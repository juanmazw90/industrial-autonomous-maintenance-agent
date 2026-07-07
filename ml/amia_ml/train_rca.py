"""
Entrena el modelo de Root Cause Analysis (RCA) de AMIA y lo registra en MLflow.

Uso:

    uv run python ml/amia_ml/train_rca.py

Variables de entorno:

    MLFLOW_TRACKING_URI : por defecto sqlite:///{repo_root}/mlflow.db
    DATA_PATH           : por defecto data/synthetic/sensor_readings.parquet

El modelo predice el modo de fallo activo (5 clases) dado un periodo de
degradación detectado por el modelo de failure prediction (V2).
Solo se entrena sobre filas con degradation_fraction > 0 — en producción
RCA se invoca únicamente cuando failure prediction ya disparó la alerta.
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
from amia_shared.features import build_features
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_PATH  = Path(os.getenv("DATA_PATH",  REPO_ROOT / "data/synthetic/sensor_readings.parquet"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

EXPERIMENT_NAME = "amia-rca"
MODEL_NAME      = "amia-rca-model"

# ── Clases objetivo ───────────────────────────────────────────────────────────
CLASSES   = sorted(["bearing_wear", "cavitation", "electrical_failure", "misalignment", "overheating"])
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

# ── Feature engineering: pipeline compartido en amia_shared.features ─────────
LEAKAGE_COLS = ["failure_mode", "is_failure", "degradation_fraction", "rul_hours", "rca_label"]
META_COLS    = ["timestamp", "machine_id", "machine_type"]

SPLIT_DATE = pd.Timestamp("2024-11-01")

# ── Hiperparámetros ───────────────────────────────────────────────────────────
HPARAMS = {
    "n_estimators":          500,
    "max_depth":             4,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "min_child_weight":      3,
    "gamma":                 0.3,
    "reg_lambda":            2.0,
    "reg_alpha":             0.5,
    "objective":             "multi:softprob",
    "num_class":             len(CLASSES),
    "eval_metric":           "mlogloss",
    "early_stopping_rounds": 30,
    "random_state":          42,
    "n_jobs":                -1,
    "verbosity":             0,
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def safe_metric(v: float) -> float:
    """Convierte NaN/Inf → 0.0 para evitar IntegrityError en SQLite de MLflow."""
    try:
        f = float(v)
        return 0.0 if (f != f or abs(f) == float("inf")) else f
    except Exception:
        return 0.0


def macro_roc_safe(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC macro tolerante a clases sin muestras en el test set."""
    per_class = []
    for i in range(y_true_bin.shape[1]):
        if y_true_bin[:, i].sum() == 0:
            per_class.append(0.0)
        else:
            per_class.append(float(roc_auc_score(y_true_bin[:, i], y_prob[:, i])))
    return float(sum(per_class) / len(per_class))


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train() -> None:
    print(f"\nMLflow tracking URI: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Cargar y preparar datos ───────────────────────────────────────────────
    print(f"\nCargando datos desde {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df_feat = build_features(df)  # incluye machine_type_enc
    df_feat["rca_label"] = df_feat["failure_mode"].map(CLASS_MAP)

    # Solo filas de degradación activa con clase válida (excluye 'normal' y 'failure')
    df_rca = df_feat[df_feat["rca_label"].notna()].copy()
    df_rca["rca_label"] = df_rca["rca_label"].astype(int)

    feature_cols = [c for c in df_rca.columns if c not in LEAKAGE_COLS + META_COLS]

    # ── Split temporal ────────────────────────────────────────────────────────
    train_mask = df_rca["timestamp"] < SPLIT_DATE
    test_mask  = df_rca["timestamp"] >= SPLIT_DATE

    X_train = df_rca.loc[train_mask, feature_cols]
    X_test  = df_rca.loc[test_mask,  feature_cols]
    y_train = df_rca.loc[train_mask, "rca_label"]
    y_test  = df_rca.loc[test_mask,  "rca_label"]

    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
    for cls, idx in CLASS_MAP.items():
        n_tr = (y_train == idx).sum()
        n_te = (y_test  == idx).sum()
        print(f"  {cls:<22} train={n_tr:>4}  test={n_te:>4}")

    # ── Balanceo ──────────────────────────────────────────────────────────────
    class_weights  = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_train.values)
    train_weights  = y_train.map(dict(enumerate(class_weights))).values

    # ── Entrenar ──────────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        print(f"\nMLflow run_id: {run.info.run_id}")

        model = xgb.XGBClassifier(
            **{k: v for k, v in HPARAMS.items() if k != "early_stopping_rounds"},
            early_stopping_rounds=HPARAMS["early_stopping_rounds"],
        )
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

        print(f"\nMejor iteración: {model.best_iteration}")
        print(f"Mejor mlogloss:  {model.best_score:.4f}")

        # ── Evaluar ───────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)

        y_test_bin = label_binarize(y_test, classes=list(range(len(CLASSES))))
        macro_roc  = macro_roc_safe(y_test_bin, y_prob)

        macro_f1    = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        print(f"\n{'='*50}")
        print(f"Macro F1:       {macro_f1:.4f}")
        print(f"Weighted F1:    {weighted_f1:.4f}")
        print(f"Macro ROC-AUC:  {macro_roc:.4f}")
        for cls in CLASSES:
            print(f"  F1 {cls:<22} {report[cls]['f1-score']:.3f}")
        print(f"{'='*50}")

        # ── Log parámetros y métricas ─────────────────────────────────────────
        mlflow.log_params({k: v for k, v in HPARAMS.items()})
        mlflow.log_params({
            "n_features":      len(feature_cols),
            "n_classes":       len(CLASSES),
            "train_rows":      len(X_train),
            "test_rows":       len(X_test),
            "split_date":      str(SPLIT_DATE),
            "class_weighting": "balanced",
        })

        mlflow.log_metric("macro_f1",       safe_metric(macro_f1))
        mlflow.log_metric("weighted_f1",    safe_metric(weighted_f1))
        mlflow.log_metric("macro_roc_auc",  safe_metric(macro_roc))
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("best_mlogloss",  model.best_score)
        for cls in CLASSES:
            mlflow.log_metric(f"f1_{cls}",       safe_metric(report[cls]["f1-score"]))
            mlflow.log_metric(f"precision_{cls}", safe_metric(report[cls]["precision"]))
            mlflow.log_metric(f"recall_{cls}",    safe_metric(report[cls]["recall"]))

        # ── Artefactos de inferencia ──────────────────────────────────────────
        tmp = tempfile.mkdtemp()

        class_map_path = f"{tmp}/rca_class_map.json"
        with open(class_map_path, "w") as f:
            json.dump(CLASS_MAP, f, indent=2)
        mlflow.log_artifact(class_map_path, artifact_path="inference")

        feature_cols_path = f"{tmp}/rca_feature_cols.json"
        with open(feature_cols_path, "w") as f:
            json.dump(list(feature_cols), f, indent=2)
        mlflow.log_artifact(feature_cols_path, artifact_path="inference")

        importances = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
        fi_path = f"{tmp}/rca_feature_importances.csv"
        importances.to_csv(fi_path, header=True)
        mlflow.log_artifact(fi_path)

        # ── Registrar modelo ──────────────────────────────────────────────────
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
