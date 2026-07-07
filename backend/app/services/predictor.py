"""
predictor.py — Servicio de predicción de fallos industriales.

Flujo de inicialización:
  1. Carga modelo desde MLflow Model Registry
  2. Descarga artefactos de inferencia (feature_cols, threshold)
  3. Lee el parquet histórico, calcula baselines y construye features iniciales
  4. Guarda el último vector de features + un buffer de lecturas raw por máquina

Flujo de predicción estática:
  Recupera el último vector de features de la máquina y puntúa en O(1).

Flujo de actualización (simulador / endpoint /sensors/reading):
  1. Recibe una nueva lectura cruda
  2. La añade al buffer de la máquina (mantiene últimas 50 lecturas)
  3. Recalcula features desde el buffer sin groupby (una sola máquina → rápido)
  4. Actualiza _latest_features y devuelve la predicción actualizada
"""

from __future__ import annotations

import json
import tempfile
import warnings
from collections import deque
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from amia_shared.features import BUFFER_SIZE, SENSORS, build_features_single

from .feature_store import FeatureBundle, load_feature_bundle

warnings.filterwarnings("ignore")


# ── Predictor ─────────────────────────────────────────────────────────────────

class FailurePredictor:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.threshold: float = 0.5
        self._latest_features: dict[str, pd.Series] = {}
        self._latest_timestamps: dict[str, str] = {}
        # buffer de lecturas raw por máquina para actualizaciones en tiempo real
        self._buffers: dict[str, deque] = {}
        # baselines precalculados: sensor → {machine_id → valor promedio}
        self._baselines: dict[str, dict[str, float]] = {}
        self.initialized: bool = False

    def initialize(
        self, mlflow_uri: str, data_path: Path, bundle: FeatureBundle | None = None
    ) -> None:
        print("[Predictor] Iniciando...")
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = "models:/amia-failure-prediction/latest"
        self.model = mlflow.xgboost.load_model(model_uri)
        print(f"[Predictor] Modelo cargado desde '{model_uri}'")

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("amia-failure-prediction")
        if not latest_versions:
            raise RuntimeError("No se encontró ninguna versión del modelo en MLflow.")
        run_id = latest_versions[0].run_id

        artifact_dir = client.download_artifacts(run_id, "inference", tempfile.mkdtemp(prefix="amia_inference_"))
        with open(f"{artifact_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{artifact_dir}/optimal_threshold.json") as f:
            self.threshold = json.load(f)["threshold"]
        print(f"[Predictor] {len(self.feature_cols)} features | umbral óptimo: {self.threshold:.3f}")

        print("[Predictor] Construyendo features para todas las máquinas...")
        if bundle is None:
            bundle = load_feature_bundle(data_path)
        df, df_feat = bundle.raw, bundle.features
        self._baselines = bundle.baselines

        for mid in df_feat["machine_id"].unique():
            rows = df_feat[df_feat["machine_id"] == mid]
            self._latest_features[mid] = rows.iloc[-1]
            self._latest_timestamps[mid] = str(rows.iloc[-1]["timestamp"])
            # Buffer inicial: últimas BUFFER_SIZE lecturas raw de cada máquina
            raw_rows = df[df["machine_id"] == mid].tail(BUFFER_SIZE)
            self._buffers[mid] = deque(
                raw_rows.to_dict("records"), maxlen=BUFFER_SIZE
            )

        self.initialized = True
        print(f"[Predictor] Listo. Máquinas: {sorted(self._latest_features.keys())}")

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, machine_id: str) -> dict:
        if not self.initialized or self.model is None:
            raise RuntimeError("El predictor no está inicializado.")

        machine_id = machine_id.upper()
        if machine_id not in self._latest_features:
            available = sorted(self._latest_features.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        latest_row = self._latest_features[machine_id]
        X = pd.DataFrame([latest_row[self.feature_cols]])
        prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "machine_id": machine_id,
            "failure_probability": round(prob, 4),
            "risk_score": round(prob, 4),
            "alert_level": self._alert_level(prob),
            "is_high_risk": prob >= self.threshold,
            "threshold_used": round(self.threshold, 3),
            "as_of_timestamp": self._latest_timestamps.get(machine_id, ""),
        }

    def predict_all(self) -> list[dict]:
        return [self.predict(mid) for mid in sorted(self._latest_features.keys())]

    # ── Actualización en tiempo real ──────────────────────────────────────────

    def update_with_reading(self, reading: dict) -> dict:
        """
        Incorpora una nueva lectura de sensores, recalcula las features de esa
        máquina y devuelve la predicción actualizada.

        Usado por POST /sensors/reading. Es síncrono y tarda ~5ms (buffer de 50 filas).
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("El predictor no está inicializado.")

        machine_id = str(reading.get("machine_id", "")).upper()
        if machine_id not in self._buffers:
            available = sorted(self._buffers.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        # Añadir al buffer (el deque descarta el más antiguo automáticamente)
        self._buffers[machine_id].append(reading)

        # Recalcular features sobre el buffer de esta máquina
        buf_df = pd.DataFrame(list(self._buffers[machine_id]))
        machine_baselines = {s: self._baselines[s].get(machine_id, 0.0) for s in SENSORS}
        feat_df = build_features_single(buf_df, machine_baselines)

        # Actualizar último vector de features
        self._latest_features[machine_id] = feat_df.iloc[-1]
        ts = reading.get("timestamp", "")
        self._latest_timestamps[machine_id] = str(ts)

        return self.predict(machine_id)

    @staticmethod
    def _alert_level(prob: float) -> str:
        if prob >= 0.7:
            return "red"
        if prob >= 0.35:
            return "yellow"
        return "green"
