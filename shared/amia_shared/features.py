"""
features.py — Pipeline de feature engineering compartido entre entrenamiento e inferencia.

Única fuente de verdad para las features de los 3 modelos (failure, RCA, RUL).
Cualquier cambio aquí afecta a entrenamiento (ml/) e inferencia (backend/) por igual:
tras modificarlo hay que reentrenar los modelos para mantener la paridad.

Entrada esperada: DataFrame ordenado por (machine_id, timestamp) con las columnas
crudas de sensores del generador sintético / SCADA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = [
    "vibration_rms",
    "vibration_peak",
    "temperature_bearing",
    "temperature_motor",
    "pressure_discharge",
    "speed_rpm",
]
WINDOWS = {"8h": 8, "24h": 24}
BASELINE_HOURS = 168   # primeras 7 días de cada máquina como referencia sana
BUFFER_SIZE = 50       # lecturas raw a mantener por máquina (inferencia en tiempo real)

MACHINE_TYPE_MAP = {"compressor": 0, "induction_motor": 1, "centrifugal_pump": 2}


def linear_slope(arr: np.ndarray) -> float:
    """Pendiente de regresión lineal sobre una ventana (ignora NaNs)."""
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(arr))
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


# ── Bloques del pipeline (multi-máquina, con groupby) ─────────────────────────

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


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_slope_{label}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=2).apply(linear_slope, raw=True)
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


# ── API pública ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo multi-máquina. El DataFrame debe venir ordenado
    por (machine_id, timestamp).
    """
    df = _add_rolling_stats(df)
    df = _add_trend_features(df)
    df = _add_delta_features(df)
    df = _add_cross_sensor_features(df)
    df = _add_baseline_normalization(df)
    df["machine_type_enc"] = df["machine_type"].map(MACHINE_TYPE_MAP)
    return df


def build_features_single(buf: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    """
    Recalcula todas las features sobre el buffer de UNA máquina (sin groupby →
    rápido para el path de tiempo real). `baselines` debe venir de
    compute_baselines() sobre el histórico: usar otro fallback rompería la
    paridad con entrenamiento.
    """
    out = buf.copy()

    for sensor in SENSORS:
        s = out[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_mean_{label}"]  = s.rolling(w, min_periods=2).mean()
            out[f"{sensor}_std_{label}"]   = s.rolling(w, min_periods=2).std()
            out[f"{sensor}_max_{label}"]   = s.rolling(w, min_periods=2).max()
            out[f"{sensor}_kurt_{label}"]  = s.rolling(w, min_periods=4).kurt()
            out[f"{sensor}_slope_{label}"] = s.rolling(w, min_periods=2).apply(linear_slope, raw=True)
        out[f"{sensor}_delta_1h"]  = s.diff(1)
        out[f"{sensor}_delta_8h"]  = s.diff(8)
        out[f"{sensor}_delta_24h"] = s.diff(24)
        out[f"{sensor}_accel"]     = s.diff(1).diff(1)
        out[f"{sensor}_vs_baseline"] = s - baselines[sensor]

    out = _add_cross_sensor_features(out)
    out["machine_type_enc"] = out["machine_type"].map(MACHINE_TYPE_MAP)
    return out


def compute_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Baselines del período sano: sensor → {machine_id → media de las primeras
    BASELINE_HOURS lecturas}. Mismo cálculo que _add_baseline_normalization.
    """
    baselines: dict[str, dict[str, float]] = {}
    for sensor in SENSORS:
        baselines[sensor] = {}
        for mid in df["machine_id"].unique():
            vals = df.loc[df["machine_id"] == mid, sensor]
            baselines[sensor][mid] = float(vals.iloc[:BASELINE_HOURS].mean())
    return baselines
