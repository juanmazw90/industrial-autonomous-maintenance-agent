"""
Etapa 1.3 — Alert engine.

AlertEngine.evaluate_predictions()
  - Reads latest model_predictions (past 5 min) from PostgreSQL.
  - Compares probability against threshold from platform_config.
  - Creates Alert row if no open/acknowledged alert already exists for that machine.
  - Publishes alert.created event to Redis.

AlertEngine.transition(alert_id, new_status, actor_id)
  - Validates state transition (new→acknowledged→assigned→resolved).
  - Updates alert row + publishes event.

start_alert_job(app)
  - Background asyncio task that runs evaluate_predictions() every 60 s.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

import structlog
from sqlalchemy import select

from app.events.publisher import publish
from app.events.schemas import AlertCreatedEvent
from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import Alert, ModelPrediction, PlatformConfig

_logger = structlog.get_logger(__name__)

AlertStatus = Literal["new", "acknowledged", "assigned", "resolved"]

_VALID_TRANSITIONS: dict[AlertStatus, list[AlertStatus]] = {
    "new":          ["acknowledged", "resolved"],
    "acknowledged": ["assigned", "resolved"],
    "assigned":     ["resolved"],
    "resolved":     [],
}

_OPEN_STATUSES = ("new", "acknowledged", "assigned")


async def _get_threshold(key: str, default: float) -> float:
    try:
        async with AsyncSessionLocal() as db:
            row = await db.get(PlatformConfig, key)
            if row and row.value and "value" in row.value:
                return float(row.value["value"])
    except Exception:
        pass
    return default


async def evaluate_predictions() -> int:
    """
    Check recent model_predictions and open alerts for machines above threshold.
    Returns the number of new alerts created.
    """
    failure_threshold = await _get_threshold("thresholds.failure", 0.85)
    cutoff = datetime.now(UTC) - timedelta(minutes=10)

    new_alerts = 0
    try:
        async with AsyncSessionLocal() as db:
            # Recent high-probability failure predictions
            preds = (await db.execute(
                select(ModelPrediction)
                .where(
                    ModelPrediction.model_name == "amia-failure-model",
                    ModelPrediction.created_at >= cutoff,
                    ModelPrediction.probability >= failure_threshold,
                )
                .order_by(ModelPrediction.created_at.desc())
            )).scalars().all()

            for pred in preds:
                # Skip if there is already an open alert for this machine
                existing = (await db.execute(
                    select(Alert).where(
                        Alert.machine_id == pred.machine_id,
                        Alert.status.in_(list(_OPEN_STATUSES)),
                    )
                )).scalar_one_or_none()

                if existing:
                    continue

                prob = round((pred.probability or 0.0) * 100, 1)
                alert_id = str(uuid.uuid4())
                alert = Alert(
                    id=alert_id,
                    machine_id=pred.machine_id,
                    severity=_severity_from_probability(pred.probability or 0.0),
                    source="failure_model",
                    title=f"Failure probability {prob}% — automatic alert",
                    description=(
                        f"Model amia-failure-model predicted {prob}% failure probability "
                        f"(threshold: {failure_threshold * 100:.0f}%). "
                        f"Prediction ID: {pred.id}"
                    ),
                    status="new",
                    prediction_id=pred.id,
                )
                db.add(alert)
                await db.flush()

                try:
                    await publish(AlertCreatedEvent(
                        payload={
                            "alert_id": alert_id,
                            "machine_id": pred.machine_id,
                            "severity": alert.severity,
                            "probability": pred.probability,
                        }
                    ))
                except Exception:
                    pass

                new_alerts += 1
                _logger.info(
                    "alert_engine: alert created",
                    machine_id=pred.machine_id,
                    probability=pred.probability,
                    alert_id=alert_id,
                )

            await db.commit()
    except Exception:
        _logger.exception("alert_engine: evaluate_predictions failed")

    return new_alerts


async def transition(
    alert_id: str,
    new_status: str,
    actor_id: str | None = None,
    assigned_to: str | None = None,
) -> Alert | None:
    """
    Apply a state transition to an alert.
    Returns the updated Alert or None if not found / invalid transition.
    """
    async with AsyncSessionLocal() as db:
        alert = await db.get(Alert, alert_id)
        if not alert:
            _logger.warning("alert_engine: alert not found", alert_id=alert_id)
            return None

        current: AlertStatus = alert.status  # type: ignore[assignment]
        allowed = _VALID_TRANSITIONS.get(current, [])
        if new_status not in allowed:
            _logger.warning(
                "alert_engine: invalid transition",
                alert_id=alert_id,
                from_status=current,
                to_status=new_status,
            )
            return None

        alert.status = new_status

        if new_status == "acknowledged" and actor_id:
            alert.acknowledged_by = actor_id
        if new_status == "assigned" and assigned_to:
            alert.assigned_to = assigned_to
        if new_status == "resolved":
            alert.resolved_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(alert)
        return alert


def _severity_from_probability(probability: float) -> str:
    if probability >= 0.95:
        return "critical"
    if probability >= 0.80:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


# ── Background job ────────────────────────────────────────────────────────────

_alert_job_running = False


async def _alert_loop() -> None:
    _logger.info("alert_engine: background job started (interval=60s)")
    while True:
        try:
            n = await evaluate_predictions()
            if n:
                _logger.info("alert_engine: created new alerts", count=n)
        except Exception:
            _logger.exception("alert_engine: job iteration failed")
        await asyncio.sleep(60)


def start_alert_job() -> asyncio.Task:
    """Schedule the alert evaluation loop as a background asyncio Task."""
    global _alert_job_running
    if _alert_job_running:
        _logger.warning("alert_engine: job already running")
    _alert_job_running = True
    return asyncio.create_task(_alert_loop(), name="alert_engine")
