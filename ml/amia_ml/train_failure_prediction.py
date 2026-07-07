"""
Entrena el modelo de predicción de fallos de AMIA y regístralo en MLflow.

Uso:

uv run python ml/amia_ml/train_failure_prediction.py

Variables de entorno:

MLFLOW_TRACKING_URI: por defecto, el directorio local ./mlruns

DATA_PATH: por defecto, data/synthetic/sensor_readings.parquet
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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

EXPERIMENT_NAME = "amia-failure-prediction"
MODEL_NAME = "amia-failure-prediction"

# ── Feature engineering: pipeline compartido en amia_shared.features ─────────
SPLIT_DATE = pd.Timestamp("2024-11-01")

LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours"]
META_COLS = ["timestamp", "machine_id", "machine_type", "label"]

# ── Hyperparameters ───────────────────────────────────────────────────────────
HPARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.5,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "max_delta_step": 1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}


# ── Training entrypoint ───────────────────────────────────────────────────────

def train() -> None:
    print(f"\nMLflow tracking URI: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Load and prepare data ─────────────────────────────────────────────────
    print(f"\nCargando datos desde {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df_feat = build_features(df.copy())  # incluye machine_type_enc
    baselines = compute_baselines(df)
    feature_cols = [c for c in df_feat.columns if c not in LEAKAGE_COLS + META_COLS]

    # Target: ¿hay fallo en las próximas 24h?
    df_feat["label"] = (
        df_feat.groupby("machine_id")["is_failure"]
        .transform(lambda x: x.astype(int).shift(-24).rolling(24, min_periods=1).max())
        .fillna(0)
        .astype(int)
    )

    # ── Temporal split ────────────────────────────────────────────────────────
    train_mask = df_feat["timestamp"] < SPLIT_DATE
    test_mask  = df_feat["timestamp"] >= SPLIT_DATE

    X_train = df_feat.loc[train_mask, feature_cols]
    X_test  = df_feat.loc[test_mask,  feature_cols]
    y_train = df_feat.loc[train_mask, "label"]
    y_test  = df_feat.loc[test_mask,  "label"]

    print(f"\nTrain: {len(X_train):,}  |  positivos: {y_train.sum():,} ({y_train.mean():.2%})")
    print(f"Test:  {len(X_test):,}   |  positivos: {y_test.sum():,} ({y_test.mean():.2%})")

    # scale_pos_weight: sqrt del ratio real × 3 (suavizado vs ratio completo)
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise RuntimeError("El split temporal no contiene positivos: ajusta SPLIT_DATE o regenera datos.")
    ratio = int((y_train == 0).sum()) / pos
    scale_pos_weight = int(np.sqrt(ratio) * 3)
    print(f"\nscale_pos_weight = {scale_pos_weight}  (ratio real = {ratio:.1f}:1)")

    # ── MLflow run ────────────────────────────────────────────────────────────
    _artifacts_dir = tempfile.mkdtemp(prefix="amia_artifacts_")
    with mlflow.start_run() as run:
        print(f"\nMLflow run_id: {run.info.run_id}")

        # Log all hyperparameters
        logged_params = {**HPARAMS, "scale_pos_weight": scale_pos_weight, "split_date": str(SPLIT_DATE)}
        mlflow.log_params(logged_params)
        mlflow.log_params({
            "n_features": len(feature_cols),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate_train": round(float(y_train.mean()), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
        })

        # ── Train ─────────────────────────────────────────────────────────────
        model = xgb.XGBClassifier(
            **{k: v for k, v in HPARAMS.items() if k != "early_stopping_rounds"},
            early_stopping_rounds=HPARAMS["early_stopping_rounds"],
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        print(f"\nMejor iteración: {model.best_iteration}")
        print(f"Mejor logloss:   {model.best_score:.4f}")
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("best_logloss_eval", model.best_score)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_prob  = model.predict_proba(X_test)[:, 1]
        pr_auc  = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Optimal threshold by F1
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_per = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = int(np.argmax(f1_per))
        best_thresh = float(thresholds[best_idx])
        best_f1 = float(f1_per[best_idx])

        y_pred_opt = (y_prob >= best_thresh).astype(int)
        report = classification_report(y_test, y_pred_opt, target_names=["normal", "fallo_24h"], output_dict=True)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("optimal_threshold", best_thresh)
        mlflow.log_metric("f1_fallo_24h", report["fallo_24h"]["f1-score"])
        mlflow.log_metric("precision_fallo_24h", report["fallo_24h"]["precision"])
        mlflow.log_metric("recall_fallo_24h", report["fallo_24h"]["recall"])

        print(f"\n{'='*50}")
        print(f"PR-AUC:           {pr_auc:.4f}")
        print(f"ROC-AUC:          {roc_auc:.4f}")
        print(f"Umbral óptimo:    {best_thresh:.3f}")
        print(f"Precision fallo:  {report['fallo_24h']['precision']:.3f}")
        print(f"Recall fallo:     {report['fallo_24h']['recall']:.3f}")
        print(f"F1 fallo:         {report['fallo_24h']['f1-score']:.3f}")
        print(f"{'='*50}")

        # ── Log artifacts ─────────────────────────────────────────────────────
        # Feature columns (needed to build the same feature vector at inference)
        feature_cols_path = os.path.join(_artifacts_dir, "feature_cols.json")
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(feature_cols_path, artifact_path="inference")

        # Baseline values per machine per sensor (needed for _vs_baseline features at inference)
        baselines_path = os.path.join(_artifacts_dir, "baselines.json")
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        mlflow.log_artifact(baselines_path, artifact_path="inference")

        # Optimal threshold
        threshold_path = os.path.join(_artifacts_dir, "optimal_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": best_thresh, "f1": best_f1}, f, indent=2)
        mlflow.log_artifact(threshold_path, artifact_path="inference")

        # Feature importances CSV
        importances = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .rename("importance")
        )
        fi_path = os.path.join(_artifacts_dir, "feature_importances.csv")
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
