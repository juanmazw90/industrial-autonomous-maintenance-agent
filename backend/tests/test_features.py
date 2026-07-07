"""
Tests de invariantes del pipeline de features compartido (amia_shared.features).

Protegen la paridad entrenamiento/inferencia: build_features (path batch de
entrenamiento e inicialización) y build_features_single (path de tiempo real)
deben producir exactamente los mismos valores sobre los mismos datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from amia_shared.features import (
    BASELINE_HOURS,
    SENSORS,
    build_features,
    build_features_single,
    compute_baselines,
)

RAW_COLS = [
    "timestamp", "machine_id", "machine_type",
    *SENSORS,
    "current_phase_a", "current_phase_b", "current_phase_c",
]

# Columnas de features que el pipeline debe generar por cada sensor
PER_SENSOR_SUFFIXES = [
    "mean_8h", "std_8h", "max_8h", "kurt_8h", "slope_8h",
    "mean_24h", "std_24h", "max_24h", "kurt_24h", "slope_24h",
    "delta_1h", "delta_8h", "delta_24h", "accel", "vs_baseline",
]
CROSS_FEATURES = [
    "current_imbalance", "temp_per_rpm", "vibro_thermal_stress",
    "temp_ratio", "current_mean", "machine_type_enc",
]


def _make_frame(n_rows: int = 250, machines: int = 2) -> pd.DataFrame:
    """Dataset sintético determinista con suficientes filas para las ventanas."""
    rng = np.random.default_rng(42)
    frames = []
    types = ["compressor", "centrifugal_pump"]
    for m in range(machines):
        ts = pd.date_range("2024-01-01", periods=n_rows, freq="h")
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "machine_id": f"MCH-{m:03d}",
            "machine_type": types[m % len(types)],
            "vibration_rms": rng.normal(2.0, 0.3, n_rows),
            "vibration_peak": rng.normal(5.0, 0.8, n_rows),
            "temperature_bearing": rng.normal(60, 5, n_rows),
            "temperature_motor": rng.normal(70, 5, n_rows),
            "pressure_discharge": rng.normal(8.0, 0.5, n_rows),
            "speed_rpm": rng.normal(2950, 30, n_rows),
            "current_phase_a": rng.normal(30, 2, n_rows),
            "current_phase_b": rng.normal(30, 2, n_rows),
            "current_phase_c": rng.normal(30, 2, n_rows),
        }))
    return pd.concat(frames, ignore_index=True)


def test_build_features_generates_expected_columns():
    df = _make_frame()
    out = build_features(df)

    expected = {f"{s}_{suffix}" for s in SENSORS for suffix in PER_SENSOR_SUFFIXES}
    expected |= set(CROSS_FEATURES)
    missing = expected - set(out.columns)
    assert not missing, f"Faltan columnas: {sorted(missing)}"
    assert len(out) == len(df)


def test_single_machine_path_matches_batch_path():
    """
    Paridad tiempo real vs batch: para una máquina, build_features_single con
    los baselines del histórico debe producir los mismos valores que
    build_features. Si esto rompe, la inferencia en vivo divergió del
    entrenamiento.
    """
    df = _make_frame(n_rows=BASELINE_HOURS + 60, machines=1)
    batch = build_features(df)

    baselines = compute_baselines(df)
    mid = df["machine_id"].iloc[0]
    machine_baselines = {s: baselines[s][mid] for s in SENSORS}
    single = build_features_single(df.copy(), machine_baselines)

    feature_cols = [
        f"{s}_{suffix}" for s in SENSORS for suffix in PER_SENSOR_SUFFIXES
    ] + CROSS_FEATURES
    pd.testing.assert_frame_equal(
        batch[feature_cols], single[feature_cols], check_exact=False, rtol=1e-12
    )


def test_compute_baselines_matches_vs_baseline_column():
    df = _make_frame()
    out = build_features(df)
    baselines = compute_baselines(df)

    for mid in df["machine_id"].unique():
        rows = out[out["machine_id"] == mid]
        for sensor in SENSORS:
            reconstructed = rows[sensor] - baselines[sensor][mid]
            pd.testing.assert_series_equal(
                rows[f"{sensor}_vs_baseline"], reconstructed,
                check_names=False, rtol=1e-12,
            )


def test_build_features_single_requires_all_baselines():
    """El fallback silencioso (media del buffer) fue eliminado a propósito:
    un baseline faltante debe fallar ruidosamente, no degradar la paridad."""
    df = _make_frame(n_rows=60, machines=1)
    with pytest.raises(KeyError):
        build_features_single(df, {})
