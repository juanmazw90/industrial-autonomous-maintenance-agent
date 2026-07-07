"""
rca_predictor.py — Servicio de diagnóstico de causa raíz (RCA).

Se invoca automáticamente por el sensor_analyst cuando FailurePredictor
detecta alert_level != "green". Responde: ¿qué modo de fallo está activo?

Flujo idéntico a FailurePredictor:
  1. Carga modelo amia-rca-model desde MLflow Model Registry
  2. Descarga artefactos de inferencia (rca_feature_cols, rca_class_map)
  3. Lee el parquet histórico, calcula baselines y construye features iniciales
  4. Guarda el último vector de features + buffer de lecturas raw por máquina

Diferencia vs FailurePredictor:
  predict() devuelve {failure_mode, confidence, probabilities} en vez de
  {failure_probability, alert_level} — el modelo es multiclase (5 clases).
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


# ── RCAPredictor ──────────────────────────────────────────────────────────────

class RCAPredictor:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.class_names: list[str] = []          # orden de clases según rca_class_map.json
        self._latest_features: dict[str, pd.Series] = {}
        self._latest_timestamps: dict[str, str] = {}
        self._buffers: dict[str, deque] = {}
        self._baselines: dict[str, dict[str, float]] = {}
        self.initialized: bool = False

    def initialize(
        self, mlflow_uri: str, data_path: Path, bundle: FeatureBundle | None = None
    ) -> None:
        print("[RCAPredictor] Iniciando...")
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = "models:/amia-rca-model/latest"
        self.model = mlflow.xgboost.load_model(model_uri)
        print(f"[RCAPredictor] Modelo cargado desde '{model_uri}'")

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("amia-rca-model")
        if not latest_versions:
            raise RuntimeError("No se encontró ninguna versión de amia-rca-model en MLflow.")
        run_id = latest_versions[0].run_id

        artifact_dir = client.download_artifacts(run_id, "inference", tempfile.mkdtemp(prefix="amia_rca_inference_"))
        with open(f"{artifact_dir}/rca_feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{artifact_dir}/rca_class_map.json") as f:
            class_map: dict[str, int] = json.load(f)
        # Invertir el mapa para obtener nombres en el orden de los índices del modelo
        self.class_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
        print(f"[RCAPredictor] {len(self.feature_cols)} features | clases: {self.class_names}")

        print("[RCAPredictor] Construyendo features para todas las máquinas...")
        if bundle is None:
            bundle = load_feature_bundle(data_path)
        df, df_feat = bundle.raw, bundle.features
        self._baselines = bundle.baselines

        for mid in df_feat["machine_id"].unique():
            rows = df_feat[df_feat["machine_id"] == mid]
            self._latest_features[mid] = rows.iloc[-1]
            self._latest_timestamps[mid] = str(rows.iloc[-1]["timestamp"])
            raw_rows = df[df["machine_id"] == mid].tail(BUFFER_SIZE)
            self._buffers[mid] = deque(raw_rows.to_dict("records"), maxlen=BUFFER_SIZE)

        self.initialized = True
        print(f"[RCAPredictor] Listo. Máquinas: {sorted(self._latest_features.keys())}")

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, machine_id: str) -> dict:
        if not self.initialized or self.model is None:
            raise RuntimeError("RCAPredictor no está inicializado.")

        machine_id = machine_id.upper()
        if machine_id not in self._latest_features:
            available = sorted(self._latest_features.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        latest_row = self._latest_features[machine_id]
        X = pd.DataFrame([latest_row[self.feature_cols]])
        proba = self.model.predict_proba(X)[0]   # shape (n_classes,) con multi:softprob
        best  = int(np.argmax(proba))

        return {
            "failure_mode":  self.class_names[best],
            "confidence":    round(float(proba[best]), 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.class_names, proba)
            },
            "as_of_timestamp": self._latest_timestamps.get(machine_id, ""),
        }

    # ── Actualización en tiempo real ──────────────────────────────────────────

    def update_with_reading(self, reading: dict) -> dict:
        if not self.initialized or self.model is None:
            raise RuntimeError("RCAPredictor no está inicializado.")

        machine_id = str(reading.get("machine_id", "")).upper()
        if machine_id not in self._buffers:
            available = sorted(self._buffers.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        self._buffers[machine_id].append(reading)
        buf_df = pd.DataFrame(list(self._buffers[machine_id]))
        machine_baselines = {s: self._baselines[s].get(machine_id, 0.0) for s in SENSORS}
        feat_df = build_features_single(buf_df, machine_baselines)

        self._latest_features[machine_id] = feat_df.iloc[-1]
        self._latest_timestamps[machine_id] = str(reading.get("timestamp", ""))

        return self.predict(machine_id)
