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
import tempfile
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

# Feature engineering: pipeline compartido (idéntico a entrenamiento e inferencia)
from amia_shared.features import build_features

_SPLIT_DATE       = pd.Timestamp("2024-11-01")
_FAILURE_MODES    = ["bearing_wear", "cavitation", "electrical_failure", "misalignment", "overheating"]
_CLASS_MAP        = {c: i for i, c in enumerate(sorted(_FAILURE_MODES))}
_CLASS_NAMES      = [k for k, _ in sorted(_CLASS_MAP.items(), key=lambda x: x[1])]


def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _test_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara temporal del test split (igual que en entrenamiento: timestamp >= 2024-11-01)."""
    return df["timestamp"] >= _SPLIT_DATE


def _download_inference_dir(model_name: str, tmp_prefix: str) -> tuple:
    """Descarga el directorio 'inference/' del run registrado y devuelve (model, artifact_dir)."""
    client   = mlflow.MlflowClient()
    model    = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
    versions = client.search_model_versions(f"name='{model_name}'")
    run_id   = versions[0].run_id
    art_dir  = client.download_artifacts(run_id, "inference", tempfile.mkdtemp(prefix=tmp_prefix))
    return model, art_dir


# ── Failure Prediction ────────────────────────────────────────────────────────

def evaluate_failure() -> dict:
    model, art_dir = _download_inference_dir("amia-failure-prediction", "fp_eval")

    with open(f"{art_dir}/feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{art_dir}/optimal_threshold.json") as f:
        threshold = json.load(f).get("threshold", 0.5)

    df_raw = _load_data()
    df_feat = build_features(df_raw)  # incluye machine_type_enc
    # Crear target igual que en train(): rolling 24h forward sobre is_failure
    df_feat["label"] = (
        df_feat.groupby("machine_id")["is_failure"]
        .transform(lambda x: x.astype(int).shift(-24).rolling(24, min_periods=1).max())
        .fillna(0).astype(int)
    )
    test = df_feat.loc[_test_mask(df_feat)].copy()

    X = test[feature_cols].fillna(0)
    y = test["label"]

    # XGBClassifier cargado con mlflow.xgboost tiene predict_proba
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "model":      "failure_prediction",
        "n_samples":  int(len(y)),
        "n_failures": int(y.sum()),
        "threshold":  round(threshold, 3),
        "auc_roc":    round(float(roc_auc_score(y, proba)), 4),
        "f1":         round(float(f1_score(y, preds, zero_division=0)), 4),
        "precision":  round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall":     round(float(recall_score(y, preds, zero_division=0)), 4),
    }


# ── RCA ───────────────────────────────────────────────────────────────────────

def evaluate_rca() -> dict:
    model, art_dir = _download_inference_dir("amia-rca-model", "rca_eval")

    with open(f"{art_dir}/rca_feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{art_dir}/rca_class_map.json") as f:
        class_map = json.load(f)    # {class_name: int_index}
    class_names  = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]

    df_raw  = _load_data()
    df_feat = build_features(df_raw)  # incluye machine_type_enc
    # Igual que train_rca.py: mapear a int y filtrar solo las 5 clases reales
    df_feat["rca_label"] = df_feat["failure_mode"].map(class_map)
    df_rca  = df_feat[df_feat["rca_label"].notna()].copy()
    df_rca["rca_label"] = df_rca["rca_label"].astype(int)
    test    = df_rca.loc[_test_mask(df_rca)].copy()

    if len(test) == 0:
        return {"model": "rca", "error": "no failure events in test split"}

    X     = test[feature_cols].fillna(0)
    y_int = test["rca_label"].values

    preds_int = model.predict(X).astype(int)
    proba     = model.predict_proba(X)

    # Etiquetas de string para métricas legibles
    y_str     = [class_names[i] if 0 <= i < len(class_names) else "unknown" for i in y_int]
    preds_str = [class_names[p] if 0 <= p < len(class_names) else "unknown" for p in preds_int]

    top3_correct = 0
    for i, true_idx in enumerate(y_int):
        top3_idx = np.argsort(proba[i])[::-1][:3]
        if true_idx in top3_idx:
            top3_correct += 1

    return {
        "model":         "rca",
        "n_samples":     int(len(y_int)),
        "accuracy":      round(float(accuracy_score(y_str, preds_str)), 4),
        "top3_accuracy": round(top3_correct / len(y_int), 4),
        "f1_macro":      round(float(f1_score(y_str, preds_str, average="macro", zero_division=0)), 4),
        "classes":       class_names,
    }


# ── RUL ──────────────────────────────────────────────────────────────────────

def evaluate_rul() -> dict:
    model, art_dir = _download_inference_dir("amia-rul-model", "rul_eval")

    with open(f"{art_dir}/rul_feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{art_dir}/rul_thresholds.json") as f:
        max_rul = json.load(f).get("max_rul_hours", 500)

    df_raw  = _load_data()
    df_feat = build_features(df_raw)
    # Solo filas etiquetadas (rul_hours no nulo), split por fecha
    labeled = df_feat[df_feat["rul_hours"].notna()].copy()
    test    = labeled.loc[_test_mask(labeled)].copy()

    X = test[feature_cols].fillna(0)
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
