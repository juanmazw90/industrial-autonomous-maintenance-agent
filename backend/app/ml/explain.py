"""
Etapa 1.2 — ML prediction persistence + SHAP explainability.

save_prediction()  — persists a ModelPrediction row and schedules SHAP
                      computation in a background task.
_compute_shap()    — runs TreeExplainer (CPU-bound) in a thread pool and
                      updates shap_top_features on the row.
MachineRegistry    — in-memory code→UUID cache loaded once at startup.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select

from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import Machine, ModelPrediction

_logger = structlog.get_logger(__name__)

# ── In-memory machine code → UUID mapping ────────────────────────────────────

_machine_registry: dict[str, str] = {}  # "PUMP-001" → UUID


async def load_machine_registry() -> None:
    """Load code→id mapping once at startup (called from lifespan)."""
    global _machine_registry
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(select(Machine.code, Machine.id))).all()
            _machine_registry = {r.code: r.id for r in rows}
            _logger.info("ml.explain: machine registry loaded", count=len(_machine_registry))
    except Exception:
        _logger.exception("ml.explain: failed to load machine registry")


def get_machine_id(code: str) -> str | None:
    """Return UUID for a machine code, or None if not found."""
    return _machine_registry.get(code.upper())


# ── SHAP background task ──────────────────────────────────────────────────────

def _run_shap_sync(
    model: Any,
    feature_row: Any,
    feature_names: list[str],
    top_n: int = 8,
) -> list[dict]:
    """
    Runs SHAP TreeExplainer synchronously (CPU-bound).
    Returns [{feature, shap_value, direction}] sorted by abs(shap_value) desc.
    """
    try:
        import numpy as np
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_row)

        # For binary classifiers shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if shap_values[1].ndim > 1 else shap_values[1]
        else:
            vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        pairs = sorted(
            zip(feature_names, vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": f, "shap_value": round(float(v), 5), "direction": "positive" if v > 0 else "negative"}
            for f, v in pairs
        ]
    except Exception as e:
        _logger.warning("ml.explain: shap computation failed", error=str(e))
        return []


async def _compute_and_save_shap(
    prediction_id: str,
    model: Any,
    feature_row: Any,
    feature_names: list[str],
) -> None:
    """Background task: compute SHAP and update the ModelPrediction row."""
    try:
        shap_features = await asyncio.to_thread(
            _run_shap_sync, model, feature_row, feature_names
        )
        if not shap_features:
            return
        async with AsyncSessionLocal() as db:
            pred = await db.get(ModelPrediction, prediction_id)
            if pred:
                pred.shap_top_features = shap_features
                await db.commit()
    except Exception:
        _logger.exception("ml.explain: compute_and_save_shap failed", prediction_id=prediction_id)


# ── Public API ────────────────────────────────────────────────────────────────

async def save_prediction(
    machine_code: str,
    model_name: str,
    model_version: str | None,
    prediction: dict,
    probability: float | None = None,
    shap_model: Any = None,
    shap_features: Any = None,
    shap_feature_names: list[str] | None = None,
) -> str | None:
    """
    Persists a ModelPrediction row and optionally schedules SHAP in background.
    Returns the prediction UUID or None on failure.
    """
    machine_id = get_machine_id(machine_code)
    if not machine_id:
        return None

    pred_id = str(uuid.uuid4())
    try:
        async with AsyncSessionLocal() as db:
            db.add(ModelPrediction(
                id=pred_id,
                machine_id=machine_id,
                model_name=model_name,
                model_version=model_version,
                prediction=prediction,
                probability=probability,
            ))
            await db.commit()
    except Exception:
        _logger.exception("ml.explain: save_prediction failed", machine=machine_code)
        return None

    if shap_model is not None and shap_features is not None and shap_feature_names:
        asyncio.create_task(
            _compute_and_save_shap(pred_id, shap_model, shap_features, shap_feature_names)
        )

    return pred_id
