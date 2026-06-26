"""
rul_predictor.py — Servicio de predicción de RUL (Remaining Useful Life).

Predice cuántas horas le quedan a una máquina antes de fallar.
Sigue el mismo patrón que FailurePredictor: inicialización desde MLflow,
buffer circular por máquina para actualizaciones en tiempo real.
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

LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS    = ["timestamp", "machine_id", "machine_type"]
MACHINE_TYPE_MAP = {"compressor": 0, "induction_motor": 1, "centrifugal_pump": 2}

_CRITICAL_HOURS = 100.0
_WARNING_HOURS  = 300.0
_MAX_RUL_HOURS  = 500.0


# ── Feature engineering (idéntico a predictor.py) ────────────────────────────

def _linear_slope(arr: np.ndarray) -> float:
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(arr))
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_mean_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=2).mean())
            out[f"{sensor}_std_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).std())
            out[f"{sensor}_max_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).max())
            out[f"{sensor}_kurt_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=4).kurt())
        for label, w in WINDOWS.items():
            out[f"{sensor}_slope_{label}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=2).apply(_linear_slope, raw=True)
            )
        out[f"{sensor}_delta_1h"]  = grp.transform(lambda x: x.diff(1))
        out[f"{sensor}_delta_8h"]  = grp.transform(lambda x: x.diff(8))
        out[f"{sensor}_delta_24h"] = grp.transform(lambda x: x.diff(24))
        out[f"{sensor}_accel"]     = grp.transform(lambda x: x.diff(1).diff(1))
    out["current_imbalance"]    = out[["current_phase_a", "current_phase_b", "current_phase_c"]].std(axis=1)
    out["temp_per_rpm"]         = out["temperature_bearing"] / out["speed_rpm"].replace(0, np.nan)
    out["vibro_thermal_stress"] = out["vibration_rms"] * out["temperature_bearing"]
    out["temp_ratio"]           = out["temperature_bearing"] / out["temperature_motor"].replace(0, np.nan)
    out["current_mean"]         = out[["current_phase_a", "current_phase_b", "current_phase_c"]].mean(axis=1)
    for sensor in SENSORS:
        baseline = out.groupby("machine_id")[sensor].transform(
            lambda x: x.iloc[:BASELINE_HOURS].mean()
        )
        out[f"{sensor}_vs_baseline"] = out[sensor] - baseline
    out["machine_type_enc"] = out["machine_type"].map(MACHINE_TYPE_MAP)
    return out


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
        out[f"{sensor}_delta_1h"]     = s.diff(1)
        out[f"{sensor}_delta_8h"]     = s.diff(8)
        out[f"{sensor}_delta_24h"]    = s.diff(24)
        out[f"{sensor}_accel"]        = s.diff(1).diff(1)
        out[f"{sensor}_vs_baseline"]  = s - baselines.get(sensor, s.mean())
    out["current_imbalance"]    = out[["current_phase_a", "current_phase_b", "current_phase_c"]].std(axis=1)
    out["temp_per_rpm"]         = out["temperature_bearing"] / out["speed_rpm"].replace(0, np.nan)
    out["vibro_thermal_stress"] = out["vibration_rms"] * out["temperature_bearing"]
    out["temp_ratio"]           = out["temperature_bearing"] / out["temperature_motor"].replace(0, np.nan)
    out["current_mean"]         = out[["current_phase_a", "current_phase_b", "current_phase_c"]].mean(axis=1)
    out["machine_type_enc"]     = out["machine_type"].map(MACHINE_TYPE_MAP)
    return out


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

    def initialize(self, mlflow_uri: str, data_path: Path) -> None:
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

        artifact_dir = client.download_artifacts(run_id, "inference", "/tmp/amia_rul_inference")
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
        feat_df = _build_features_single(buf_df, machine_baselines)

        self._latest_features[machine_id]   = feat_df.iloc[-1]
        self._latest_timestamps[machine_id] = str(reading.get("timestamp", ""))
        return self.predict(machine_id)

    def _urgency(self, hours: float) -> str:
        if hours < self.critical_hours:
            return "critical"
        if hours < self.warning_hours:
            return "warning"
        return "normal"
