"""
predictor.py — Servicio de predicción de fallos industriales.

Carga el modelo XGBoost desde MLflow al arrancar la app, precalcula
las features para todas las máquinas y sirve predicciones en O(1).

Flujo de inicialización:
  1. Carga modelo desde MLflow Model Registry
  2. Descarga artefactos de inferencia (feature_cols, baselines, threshold)
  3. Lee el parquet histórico y construye features para todas las máquinas
  4. Guarda el último vector de features por máquina en memoria

Flujo de predicción (una vez inicializado):
  1. Recupera el último vector de features de la máquina
  2. Puntúa con el modelo y devuelve probabilidad + nivel de alerta
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Constantes de feature engineering (deben coincidir con el script de entrenamiento) ──
SENSORS = [
    "vibration_rms",
    "vibration_peak",
    "temperature_bearing",
    "temperature_motor",
    "pressure_discharge",
    "speed_rpm",
]
WINDOWS = {"8h": 8, "24h": 24}
BASELINE_HOURS = 168

LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS = ["timestamp", "machine_id", "machine_type", "label"]
MACHINE_TYPE_MAP = {"compressor": 0, "induction_motor": 1, "centrifugal_pump": 2}


# ── Feature engineering ───────────────────────────────────────────────────────

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


# ── Predictor ─────────────────────────────────────────────────────────────────

class FailurePredictor:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.threshold: float = 0.5
        self._latest_features: dict[str, pd.Series] = {}
        self._latest_timestamps: dict[str, str] = {}
        self.initialized: bool = False

    def initialize(self, mlflow_uri: str, data_path: Path) -> None:
        """
        Carga el modelo desde MLflow y precalcula las features para todas las máquinas.
        Se llama una vez al arrancar la app (lifespan handler).
        """
        print("[Predictor] Iniciando...")
        mlflow.set_tracking_uri(mlflow_uri)

        # Cargar modelo desde Model Registry
        model_uri = "models:/amia-failure-prediction/latest"
        self.model = mlflow.xgboost.load_model(model_uri)
        print(f"[Predictor] Modelo cargado desde '{model_uri}'")

        # Descargar artefactos de inferencia
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("amia-failure-prediction")
        if not latest_versions:
            raise RuntimeError("No se encontró ninguna versión del modelo en MLflow.")
        run_id = latest_versions[0].run_id

        artifact_dir = client.download_artifacts(run_id, "inference", "/tmp/amia_inference")
        with open(f"{artifact_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)
        with open(f"{artifact_dir}/optimal_threshold.json") as f:
            self.threshold = json.load(f)["threshold"]
        print(f"[Predictor] {len(self.feature_cols)} features | umbral óptimo: {self.threshold:.3f}")

        # Precalcular features para todas las máquinas
        print("[Predictor] Construyendo features para todas las máquinas...")
        df = pd.read_parquet(data_path)
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        df_feat = _build_features(df)

        for machine_id in df_feat["machine_id"].unique():
            machine_rows = df_feat[df_feat["machine_id"] == machine_id]
            latest = machine_rows.iloc[-1]
            self._latest_features[machine_id] = latest
            self._latest_timestamps[machine_id] = str(latest["timestamp"])

        self.initialized = True
        machines = list(self._latest_features.keys())
        print(f"[Predictor] Listo. Máquinas disponibles: {machines}")

    def predict(self, machine_id: str) -> dict:
        """
        Devuelve la probabilidad de fallo en las próximas 24h para una máquina.

        Returns:
            machine_id, failure_probability, risk_score, alert_level,
            is_high_risk, threshold_used, as_of_timestamp
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("El predictor no está inicializado.")

        machine_id = machine_id.upper()
        if machine_id not in self._latest_features:
            available = sorted(self._latest_features.keys())
            raise ValueError(f"Máquina '{machine_id}' no encontrada. Disponibles: {available}")

        latest_row = self._latest_features[machine_id]
        X = pd.DataFrame([latest_row[self.feature_cols]])
        prob = float(self.model.predict_proba(X)[0, 1])

        if prob >= 0.7:
            alert_level = "red"
        elif prob >= 0.35:
            alert_level = "yellow"
        else:
            alert_level = "green"

        return {
            "machine_id": machine_id,
            "failure_probability": round(prob, 4),
            "risk_score": round(prob, 4),
            "alert_level": alert_level,
            "is_high_risk": prob >= self.threshold,
            "threshold_used": round(self.threshold, 3),
            "as_of_timestamp": self._latest_timestamps.get(machine_id, ""),
        }

    def predict_all(self) -> list[dict]:
        """Devuelve predicciones para todas las máquinas conocidas."""
        return [self.predict(mid) for mid in sorted(self._latest_features.keys())]
