"""
evaluate.py — Suite de evaluación formal de modelos AMIA.

Carga los modelos registrados en MLflow y evalúa sobre el test split (20% temporal):
  - Failure prediction: AUC-ROC, F1, Precision, Recall @ umbral óptimo
  - RCA multiclase:     Accuracy, Top-3 Accuracy, F1-macro
  - RUL regression:     RMSE, MAE, R²

Uso:
  uv run python ml/amia_ml/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT   = Path(__file__).resolve().parents[2]
MLFLOW_URI  = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DATA_PATH   = REPO_ROOT / "data/synthetic/sensor_readings.parquet"
RESULTS_DIR = REPO_ROOT / "data/evaluation"

mlflow.set_tracking_uri(MLFLOW_URI)


def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _feature_engineering(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Replica el feature engineering de los scripts de entrenamiento."""
    df = df.copy()
    for col in ["vibration_rms", "vibration_peak", "temperature_bearing", "temperature_motor",
                "pressure_discharge", "current_phase_a", "current_phase_b", "current_phase_c", "speed_rpm"]:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"]    = df.groupby("machine_id")[col].shift(1)
        df[f"{col}_roll3"]   = df.groupby("machine_id")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll6"]   = df.groupby("machine_id")[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df[f"{col}_std3"]    = df.groupby("machine_id")[col].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df[f"{col}_delta"]   = df.groupby("machine_id")[col].diff().fillna(0)

    if "machine_type" in df.columns:
        df = pd.get_dummies(df, columns=["machine_type"], prefix="mtype")

    return df[feature_cols].fillna(0) if feature_cols else df


def _test_split(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el 20% final del dataset (split temporal, igual que en entrenamiento)."""
    cutoff = int(len(df) * 0.8)
    return df.iloc[cutoff:].copy()


# ── Failure Prediction ────────────────────────────────────────────────────────

def evaluate_failure() -> dict:
    client = mlflow.MlflowClient()
    model  = mlflow.sklearn.load_model("models:/amia-failure-prediction/latest")

    # Cargar artefactos
    run_id = client.get_latest_versions("amia-failure-prediction")[0].run_id
    art_path = client.download_artifacts(run_id, "feature_cols.json", "/tmp")
    with open(art_path) as f:
        feature_cols = json.load(f)
    art_path2 = client.download_artifacts(run_id, "thresholds.json", "/tmp")
    with open(art_path2) as f:
        thresholds = json.load(f)
    threshold = thresholds.get("optimal", 0.5)

    df   = _load_data()
    test = _test_split(df)
    X    = _feature_engineering(test, feature_cols)
    y    = test["will_fail_24h"].fillna(0).astype(int)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "model":     "failure_prediction",
        "n_samples": int(len(y)),
        "n_failures": int(y.sum()),
        "threshold": round(threshold, 3),
        "auc_roc":   round(float(roc_auc_score(y, proba)), 4),
        "f1":        round(float(f1_score(y, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, preds, zero_division=0)), 4),
    }


# ── RCA ───────────────────────────────────────────────────────────────────────

def evaluate_rca() -> dict:
    client = mlflow.MlflowClient()
    model  = mlflow.sklearn.load_model("models:/amia-rca-model/latest")

    run_id   = client.get_latest_versions("amia-rca-model")[0].run_id
    art_path = client.download_artifacts(run_id, "feature_cols.json", "/tmp/rca_")
    with open(art_path) as f:
        feature_cols = json.load(f)

    df   = _load_data()
    test = _test_split(df)
    test = test[test["failure_mode"].notna()].copy()

    if len(test) == 0:
        return {"model": "rca", "error": "no failure events in test split"}

    X  = _feature_engineering(test, feature_cols)
    y  = test["failure_mode"].astype(str)

    preds = model.predict(X)
    proba = model.predict_proba(X)  # shape (n, n_classes)

    # Top-3 accuracy
    top3_correct = 0
    classes = list(model.classes_)
    for i, true_label in enumerate(y):
        top3_idx = np.argsort(proba[i])[::-1][:3]
        top3_labels = [classes[j] for j in top3_idx]
        if true_label in top3_labels:
            top3_correct += 1

    return {
        "model":        "rca",
        "n_samples":    int(len(y)),
        "accuracy":     round(float(accuracy_score(y, preds)), 4),
        "top3_accuracy": round(top3_correct / len(y), 4),
        "f1_macro":     round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
    }


# ── RUL ──────────────────────────────────────────────────────────────────────

def evaluate_rul() -> dict:
    client = mlflow.MlflowClient()
    model  = mlflow.sklearn.load_model("models:/amia-rul-model/latest")

    run_id   = client.get_latest_versions("amia-rul-model")[0].run_id
    art_path = client.download_artifacts(run_id, "rul_feature_cols.json", "/tmp/rul_")
    with open(art_path) as f:
        feature_cols = json.load(f)
    art_path2 = client.download_artifacts(run_id, "rul_baselines.json", "/tmp/rul_")
    with open(art_path2) as f:
        baselines = json.load(f)
    max_rul = baselines.get("max_rul_hours", 500)

    df   = _load_data()
    test = _test_split(df[df["rul_hours"].notna()].copy())

    X = _feature_engineering(test, feature_cols)
    y = test["rul_hours"].values

    raw   = model.predict(X)
    preds = np.clip(raw, 0, max_rul)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae  = float(mean_absolute_error(y, preds))
    r2   = float(r2_score(y, preds))

    return {
        "model":     "rul",
        "n_samples": int(len(y)),
        "rmse":      round(rmse, 2),
        "mae":       round(mae, 2),
        "r2":        round(r2, 4),
        "max_rul_hours": max_rul,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all() -> dict:
    results: dict[str, dict] = {}

    print("🔍 Evaluando failure prediction…")
    try:
        results["failure_prediction"] = evaluate_failure()
        fp = results["failure_prediction"]
        print(f"   AUC-ROC: {fp['auc_roc']}  F1: {fp['f1']}  Precision: {fp['precision']}  Recall: {fp['recall']}")
    except Exception as e:
        results["failure_prediction"] = {"error": str(e)}
        print(f"   ERROR: {e}")

    print("🔍 Evaluando RCA…")
    try:
        results["rca"] = evaluate_rca()
        rca = results["rca"]
        print(f"   Accuracy: {rca['accuracy']}  Top-3: {rca['top3_accuracy']}  F1-macro: {rca['f1_macro']}")
    except Exception as e:
        results["rca"] = {"error": str(e)}
        print(f"   ERROR: {e}")

    print("🔍 Evaluando RUL…")
    try:
        results["rul"] = evaluate_rul()
        r = results["rul"]
        print(f"   RMSE: {r['rmse']}h  MAE: {r['mae']}h  R²: {r['r2']}")
    except Exception as e:
        results["rul"] = {"error": str(e)}
        print(f"   ERROR: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "evaluation_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✅ Resultados guardados en {out}")
    return results


if __name__ == "__main__":
    run_all()
