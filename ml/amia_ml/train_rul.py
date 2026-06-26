"""
Entrena el modelo de predicción de RUL (Remaining Useful Life) de AMIA.

Uso:
    uv run python ml/amia_ml/train_rul.py

Solo entrena sobre filas con rul_hours no nulo (período de degradación activa).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH  = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

EXPERIMENT_NAME = "amia-rul-prediction"
MODEL_NAME      = "amia-rul-model"

SENSORS = [
    "vibration_rms",
    "vibration_peak",
    "temperature_bearing",
    "temperature_motor",
    "pressure_discharge",
    "speed_rpm",
]
WINDOWS        = {"8h": 8, "24h": 24}
SPLIT_DATE     = pd.Timestamp("2024-11-01")
BASELINE_HOURS = 168
MAX_RUL_HOURS  = 500.0   # coincide con MAX_RUL_HORIZON del generator

LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS    = ["timestamp", "machine_id", "machine_type"]

MACHINE_TYPE_MAP = {"compressor": 0, "induction_motor": 1, "centrifugal_pump": 2}

HPARAMS = {
    "n_estimators":      500,
    "max_depth":         4,
    "learning_rate":     0.03,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "min_child_weight":  5,
    "gamma":             0.3,
    "reg_lambda":        3.0,
    "reg_alpha":         1.0,
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
    "early_stopping_rounds": 50,
    "random_state":      42,
    "n_jobs":            -1,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def _linear_slope(arr: np.ndarray) -> float:
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(arr))
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    print("1/4  Rolling stats (mean / std / max / kurt × 8h y 24h)...")
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_mean_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=2).mean())
            out[f"{sensor}_std_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).std())
            out[f"{sensor}_max_{label}"]  = grp.transform(lambda x: x.rolling(w, min_periods=2).max())
            out[f"{sensor}_kurt_{label}"] = grp.transform(lambda x: x.rolling(w, min_periods=4).kurt())

    print("2/4  Tendencias (pendiente lineal) y deltas...")
    for sensor in SENSORS:
        grp = out.groupby("machine_id")[sensor]
        for label, w in WINDOWS.items():
            out[f"{sensor}_slope_{label}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=2).apply(_linear_slope, raw=True)
            )
        out[f"{sensor}_delta_1h"]  = grp.transform(lambda x: x.diff(1))
        out[f"{sensor}_delta_8h"]  = grp.transform(lambda x: x.diff(8))
        out[f"{sensor}_delta_24h"] = grp.transform(lambda x: x.diff(24))
        out[f"{sensor}_accel"]     = grp.transform(lambda x: x.diff(1).diff(1))

    print("3/4  Cross-sensor features...")
    out["current_imbalance"]    = out[["current_phase_a", "current_phase_b", "current_phase_c"]].std(axis=1)
    out["temp_per_rpm"]         = out["temperature_bearing"] / out["speed_rpm"].replace(0, np.nan)
    out["vibro_thermal_stress"] = out["vibration_rms"] * out["temperature_bearing"]
    out["temp_ratio"]           = out["temperature_bearing"] / out["temperature_motor"].replace(0, np.nan)
    out["current_mean"]         = out[["current_phase_a", "current_phase_b", "current_phase_c"]].mean(axis=1)

    print("4/4  Baseline normalization por máquina...")
    baselines: dict[str, dict[str, float]] = {}
    for sensor in SENSORS:
        sensor_baselines = (
            out.groupby("machine_id")[sensor]
            .apply(lambda x: x.iloc[:BASELINE_HOURS].mean())
            .to_dict()
        )
        baselines[sensor] = sensor_baselines
        baseline_series = out.groupby("machine_id")[sensor].transform(
            lambda x: x.iloc[:BASELINE_HOURS].mean()
        )
        out[f"{sensor}_vs_baseline"] = out[sensor] - baseline_series

    out["machine_type_enc"] = out["machine_type"].map(MACHINE_TYPE_MAP)
    print(f"     Shape final: {out.shape}")
    return out, baselines


# ── Training entrypoint ───────────────────────────────────────────────────────

def train() -> None:
    print(f"\nMLflow tracking URI: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nCargando datos desde {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df_feat, baselines = build_features(df.copy())

    feature_cols = [c for c in df_feat.columns if c not in LEAKAGE_COLS + META_COLS]

    # Solo filas con rul_hours etiquetado (período de degradación activa)
    labeled = df_feat[df_feat["rul_hours"].notna()].copy()
    print(f"\nFilas etiquetadas (rul_hours no nulo): {len(labeled):,}")
    print(
        f"RUL — media: {labeled['rul_hours'].mean():.1f}h  "
        f"min: {labeled['rul_hours'].min():.0f}h  "
        f"max: {labeled['rul_hours'].max():.0f}h"
    )

    train_mask = labeled["timestamp"] < SPLIT_DATE
    test_mask  = labeled["timestamp"] >= SPLIT_DATE

    X_train = labeled.loc[train_mask, feature_cols]
    X_test  = labeled.loc[test_mask,  feature_cols]
    y_train = labeled.loc[train_mask, "rul_hours"]
    y_test  = labeled.loc[test_mask,  "rul_hours"]

    print(f"\nTrain: {len(X_train):,} filas  |  Test: {len(X_test):,} filas")

    with mlflow.start_run() as run:
        print(f"\nMLflow run_id: {run.info.run_id}")

        mlflow.log_params({
            **{k: v for k, v in HPARAMS.items() if k != "early_stopping_rounds"},
            "split_date":  str(SPLIT_DATE),
            "n_features":  len(feature_cols),
            "train_rows":  len(X_train),
            "test_rows":   len(X_test),
            "max_rul_hours": MAX_RUL_HOURS,
        })

        # ── Train ─────────────────────────────────────────────────────────────
        model = xgb.XGBRegressor(
            **{k: v for k, v in HPARAMS.items() if k != "early_stopping_rounds"},
            early_stopping_rounds=HPARAMS["early_stopping_rounds"],
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = np.clip(model.predict(X_test), 0, MAX_RUL_HOURS)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("best_iteration", model.best_iteration)

        print(f"\n{'='*50}")
        print(f"RMSE:            {rmse:.2f}h")
        print(f"MAE:             {mae:.2f}h")
        print(f"Mejor iteración: {model.best_iteration}")
        print(f"{'='*50}")

        # ── Log artifacts ─────────────────────────────────────────────────────
        feature_cols_path = "/tmp/rul_feature_cols.json"
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(feature_cols_path, artifact_path="inference")

        baselines_path = "/tmp/rul_baselines.json"
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        mlflow.log_artifact(baselines_path, artifact_path="inference")

        thresholds_path = "/tmp/rul_thresholds.json"
        with open(thresholds_path, "w") as f:
            json.dump({
                "max_rul_hours":   MAX_RUL_HOURS,
                "critical_hours":  100,
                "warning_hours":   300,
                "rmse":            rmse,
                "mae":             mae,
            }, f, indent=2)
        mlflow.log_artifact(thresholds_path, artifact_path="inference")

        importances = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .rename("importance")
        )
        fi_path = "/tmp/rul_feature_importances.csv"
        importances.to_csv(fi_path, header=True)
        mlflow.log_artifact(fi_path)

        # ── Register model ────────────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"\nModelo registrado como '{MODEL_NAME}' en MLflow.")
        print(f"Run ID: {run.info.run_id}")
        print(f"\nTop 10 features:")
        print(importances.head(10).round(4).to_string())


if __name__ == "__main__":
    train()
