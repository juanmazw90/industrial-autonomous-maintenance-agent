"""
Entrena el modelo de predicción de RUL (Remaining Useful Life) de AMIA.

Uso:
    uv run python ml/amia_ml/train_rul.py

Solo entrena sobre filas con rul_hours no nulo (período de degradación activa).
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from amia_shared.features import build_features, compute_baselines
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH  = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

EXPERIMENT_NAME = "amia-rul-prediction"
MODEL_NAME      = "amia-rul-model"

# Feature engineering: pipeline compartido en amia_shared.features
SPLIT_DATE    = pd.Timestamp("2024-11-01")
MAX_RUL_HOURS = 500.0   # coincide con MAX_RUL_HORIZON del generator

LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS    = ["timestamp", "machine_id", "machine_type"]

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


# ── Training entrypoint ───────────────────────────────────────────────────────

def train() -> None:
    print(f"\nMLflow tracking URI: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nCargando datos desde {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df_feat = build_features(df.copy())  # incluye machine_type_enc
    baselines = compute_baselines(df)

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

    _artifacts_dir = tempfile.mkdtemp(prefix="amia_artifacts_")
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
        feature_cols_path = os.path.join(_artifacts_dir, "rul_feature_cols.json")
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(feature_cols_path, artifact_path="inference")

        baselines_path = os.path.join(_artifacts_dir, "rul_baselines.json")
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        mlflow.log_artifact(baselines_path, artifact_path="inference")

        thresholds_path = os.path.join(_artifacts_dir, "rul_thresholds.json")
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
        fi_path = os.path.join(_artifacts_dir, "rul_feature_importances.csv")
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
        print("\nTop 10 features:")
        print(importances.head(10).round(4).to_string())


if __name__ == "__main__":
    train()
