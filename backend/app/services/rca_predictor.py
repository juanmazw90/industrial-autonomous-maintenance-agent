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
import warnings
from collections import deque
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

SENSORS = [
    "vibration_rms",
    "vibration_peak",
    "temperature_bearing",
    "temperature_motor",
    "pressure_discharge",
    "speed_rpm",
]
WINDOWS        = {"8h": 8, "24h": 24}
BASELINE_HOURS = 168
BUFFER_SIZE    = 50

LEAKAGE_COLS     = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS        = ["timestamp", "machine_id", "machine_type"]
MACHINE_TYPE_MAP = {"compressor": 0, "induction_motor": 1, "centrifugal_pump": 2}


# ── Feature engineering sobre múltiples máquinas (inicialización) ─────────────

def _add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_mean_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=2).mean())
            out[f"{sensor}_std_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).std())
            out[f"{sensor}_max_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).max())
            out[f"{sensor}_kurt_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=4).kurt())
    return out


def _linear_slope(arr: np.ndarray) -> float:
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(arr))
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_slope_{label}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=2).apply(_linear_slope, raw=True)
            )
    return out


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        out[f"{sensor}_delta_1h"]  = grp.transform(lambda x: x.diff(1))
        out[f"{sensor}_delta_8h"]  = grp.transform(lambda x: x.diff(8))
        out[f"{sensor}_delta_24h"] = grp.transform(lambda x: x.diff(24))
        out[f"{sensor}_accel"]     = grp.transform(lambda x: x.diff(1).diff(1))
    return out


def _add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["current_imbalance"]    = out[["current_phase_a", "current_phase_b", "current_phase_c"]].std(axis=1)
    out["temp_per_rpm"]         = out["temperature_bearing"] / out["speed_rpm"].replace(0, np.nan)
    out["vibro_thermal_stress"] = out["vibration_rms"] * out["temperature_bearing"]
    out["temp_ratio"]           = out["temperature_bearing"] / out["temperature_motor"].replace(0, np.nan)
    out["current_mean"]         = out[["current_phase_a", "current_phase_b", "current_phase_c"]].mean(axis=1)
    return out


def _add_baseline_normalization(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        baseline = out.groupby("machine_id")[sensor].transform(
            lambda x: x.iloc[:BASELINE_HOURS].mean()
        )
        out[f"{sensor}_vs_baseline"] = out[sensor] - baseline
    return out


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_rolling_stats(df)
    df = _add_trend_features(df)
    df = _add_delta_features(df)
    df = _add_cross_sensor_features(df)
    df = _add_baseline_normalization(df)
    df["machine_type_enc"] = df["machine_type"].map(MACHINE_TYPE_MAP)
    return df


# ── Feature engineering sobre una sola máquina (actualización en tiempo real) ──

def _build_features_single(buf: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    out = buf.copy()
    for sensor in SENSORS:
        s = out[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_mean_{label}"]  = s.rolling(w, min_periods=2).mean()
            out[f"{sensor}_std_{label}"]   = s.rolling(w, min_periods=2).std()
            out[f"{sensor}_max_{label}"]   = s.rolling(w, min_periods=2).max()
            out[f"{sensor}_kurt_{label}"]  = s.rolling(w, min_periods=4).kurt()
            out[f"{sensor}_slope_{label}"] = s.rolling(w, min_periods=2).apply(_linear_slope, raw=True)
        out[f"{sensor}_delta_1h"]  = s.diff(1)
        out[f"{sensor}_delta_8h"]  = s.diff(8)
        out[f"{sensor}_delta_24h"] = s.diff(24)
        out[f"{sensor}_accel"]     = s.diff(1).diff(1)
        out[f"{sensor}_vs_baseline"] = s - baselines.get(sensor, s.mean())

    out["current_imbalance"]    = out[["current_phase_a", "current_phase_b", "current_phase_c"]].std(axis=1)
    out["temp_per_rpm"]         = out["temperature_bearing"] / out["speed_rpm"].replace(0, np.nan)
    out["vibro_thermal_stress"] = out["vibration_rms"] * out["temperature_bearing"]
    out["temp_ratio"]           = out["temperature_bearing"] / out["temperature_motor"].replace(0, np.nan)
    out["current_mean"]         = out[["current_phase_a", "current_phase_b", "current_phase_c"]].mean(axis=1)
    out["machine_type_enc"]     = out["machine_type"].map(MACHINE_TYPE_MAP)
    return out


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

    def initialize(self, mlflow_uri: str, data_path: Path) -> None:
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

        artifact_dir = client.download_artifacts(run_id, "inference", "/tmp/amia_rca_inference")
        with open(f"{artifact_dir}/rca_feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{artifact_dir}/rca_class_map.json") as f:
            class_map: dict[str, int] = json.load(f)
        # Invertir el mapa para obtener nombres en el orden de los índices del modelo
        self.class_names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
        print(f"[RCAPredictor] {len(self.feature_cols)} features | clases: {self.class_names}")

        print("[RCAPredictor] Construyendo features para todas las máquinas...")
        df = pd.read_parquet(data_path)
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

        for sensor in SENSORS:
            self._baselines[sensor] = {}
            for mid in df["machine_id"].unique():
                vals = df.loc[df["machine_id"] == mid, sensor]
                self._baselines[sensor][mid] = float(vals.iloc[:BASELINE_HOURS].mean())

        df_feat = _build_features(df)

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
        feat_df = _build_features_single(buf_df, machine_baselines)

        self._latest_features[machine_id] = feat_df.iloc[-1]
        self._latest_timestamps[machine_id] = str(reading.get("timestamp", ""))

        return self.predict(machine_id)
