"""
rul_predictor.py — Servicio de predicción de RUL (Remaining Useful Life).

Predice cuántas horas le quedan a una máquina antes de fallar.
Sigue el mismo patrón que FailurePredictor: inicialización desde MLflow,
buffer circular por máquina para actualizaciones en tiempo real.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from collections import deque
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from amia_shared.features import BUFFER_SIZE, SENSORS, build_features_single

from .feature_store import FeatureBundle, load_feature_bundle

warnings.filterwarnings("ignore")

_CRITICAL_HOURS = 100.0
_WARNING_HOURS  = 300.0
_MAX_RUL_HOURS  = 500.0


# ── RULPredictor ──────────────────────────────────────────────────────────────

class RULPredictor:
    def __init__(self) -> None:
        self.model: xgb.XGBRegressor | None = None
        self.feature_cols: list[str] = []
        self.critical_hours: float = _CRITICAL_HOURS
        self.warning_hours: float  = _WARNING_HOURS
        self.max_rul_hours: float  = _MAX_RUL_HOURS
        self._latest_features: dict[str, pd.Series] = {}
        self._latest_timestamps: dict[str, str] = {}
        self._buffers: dict[str, deque] = {}
        self._baselines: dict[str, dict[str, float]] = {}
        self.initialized: bool = False

    def initialize(
        self, mlflow_uri: str, data_path: Path, bundle: FeatureBundle | None = None
    ) -> None:
        print("[RULPredictor] Iniciando...")
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = "models:/amia-rul-model/latest"
        self.model = mlflow.xgboost.load_model(model_uri)
        print(f"[RULPredictor] Modelo cargado desde '{model_uri}'")

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("amia-rul-model")
        if not latest_versions:
            raise RuntimeError("No se encontró ninguna versión del modelo RUL en MLflow.")
        run_id = latest_versions[0].run_id

        artifact_dir = client.download_artifacts(run_id, "inference", tempfile.mkdtemp(prefix="amia_rul_inference_"))
        with open(f"{artifact_dir}/rul_feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{artifact_dir}/rul_thresholds.json") as f:
            thr = json.load(f)
            self.critical_hours = thr.get("critical_hours", _CRITICAL_HOURS)
            self.warning_hours  = thr.get("warning_hours", _WARNING_HOURS)
            self.max_rul_hours  = thr.get("max_rul_hours", _MAX_RUL_HOURS)

        print(
            f"[RULPredictor] {len(self.feature_cols)} features | "
            f"crítico <{self.critical_hours:.0f}h | alerta <{self.warning_hours:.0f}h"
        )

        print("[RULPredictor] Construyendo features para todas las máquinas...")
        if bundle is None:
            bundle = load_feature_bundle(data_path)
        df, df_feat = bundle.raw, bundle.features
        self._baselines = bundle.baselines

        for mid in df_feat["machine_id"].unique():
            rows = df_feat[df_feat["machine_id"] == mid]
            self._latest_features[mid]    = rows.iloc[-1]
            self._latest_timestamps[mid]  = str(rows.iloc[-1]["timestamp"])
            raw_rows = df[df["machine_id"] == mid].tail(BUFFER_SIZE)
            self._buffers[mid] = deque(raw_rows.to_dict("records"), maxlen=BUFFER_SIZE)

        self.initialized = True
        print(f"[RULPredictor] Listo. Máquinas: {sorted(self._latest_features.keys())}")

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, machine_id: str) -> dict:
        if not self.initialized or self.model is None:
            raise RuntimeError("El RULPredictor no está inicializado.")

        machine_id = machine_id.upper()
        if machine_id not in self._latest_features:
            available = sorted(self._latest_features.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        latest_row  = self._latest_features[machine_id]
        X           = pd.DataFrame([latest_row[self.feature_cols]])
        raw_pred    = float(self.model.predict(X)[0])
        hours       = float(np.clip(raw_pred, 0.0, self.max_rul_hours))
        degradation = round(1.0 - hours / self.max_rul_hours, 3)

        return {
            "machine_id":          machine_id,
            "hours_remaining":     round(hours, 1),
            "degradation_fraction": degradation,
            "urgency_level":       self._urgency(hours),
            "as_of_timestamp":     self._latest_timestamps.get(machine_id, ""),
        }

    def predict_all(self) -> list[dict]:
        return [self.predict(mid) for mid in sorted(self._latest_features.keys())]

    # ── Actualización en tiempo real ──────────────────────────────────────────

    def update_with_reading(self, reading: dict) -> dict:
        if not self.initialized or self.model is None:
            raise RuntimeError("El RULPredictor no está inicializado.")

        machine_id = str(reading.get("machine_id", "")).upper()
        if machine_id not in self._buffers:
            available = sorted(self._buffers.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        self._buffers[machine_id].append(reading)
        buf_df = pd.DataFrame(list(self._buffers[machine_id]))
        machine_baselines = {s: self._baselines[s].get(machine_id, 0.0) for s in SENSORS}
        feat_df = build_features_single(buf_df, machine_baselines)

        self._latest_features[machine_id]   = feat_df.iloc[-1]
        self._latest_timestamps[machine_id] = str(reading.get("timestamp", ""))
        return self.predict(machine_id)

    def _urgency(self, hours: float) -> str:
        if hours < self.critical_hours:
            return "critical"
        if hours < self.warning_hours:
            return "warning"
        return "normal"
