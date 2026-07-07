"""
monitor_drift.py — ML monitoring con Evidently AI.

Detecta data drift comparando:
  - Referencia: primer 60% del dataset (producción histórica)
  - Actual:     último 20% del dataset (producción reciente)

Genera:
  - data/drift_report.html     — reporte visual interactivo
  - data/drift_summary.json    — resumen JSON para el endpoint /metrics/drift

Uso:
  uv run python ml/amia_ml/monitor_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_PATH   = REPO_ROOT / "data/synthetic/sensor_readings.parquet"
OUTPUT_DIR  = REPO_ROOT / "data"

SENSOR_COLS = [
    "vibration_rms", "vibration_peak",
    "temperature_bearing", "temperature_motor",
    "pressure_discharge",
    "current_phase_a", "current_phase_b", "current_phase_c",
    "speed_rpm",
]

TARGET_COL  = "is_failure"
MACHINE_COL = "machine_id"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n      = len(df)
    ref    = df.iloc[: int(n * 0.6)].copy()
    curr   = df.iloc[int(n * 0.8) :].copy()
    return ref, curr


def run_drift_report() -> dict:
    ref, curr = load_splits()

    cols_to_use = [c for c in SENSOR_COLS if c in ref.columns] + [TARGET_COL]

    column_mapping = ColumnMapping(
        target=TARGET_COL,
        numerical_features=[c for c in SENSOR_COLS if c in ref.columns],
    )

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref[cols_to_use], current_data=curr[cols_to_use], column_mapping=column_mapping)

    # Guardar HTML
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / "drift_report.html"
    report.save_html(str(html_path))

    # Extraer resumen JSON
    report_dict = report.as_dict()
    metrics_raw = report_dict.get("metrics", [])

    drifted_features: list[str] = []
    feature_drift_scores: dict[str, float] = {}

    for m in metrics_raw:
        result = m.get("result", {})
        # DataDriftPreset expone drift_by_columns
        drift_by_col = result.get("drift_by_columns", {})
        for col, info in drift_by_col.items():
            score = info.get("drift_score", 0.0)
            feature_drift_scores[col] = round(float(score), 4)
            if info.get("drift_detected", False):
                drifted_features.append(col)

    dataset_drift = len(drifted_features) > 0

    summary = {
        "reference_rows":    len(ref),
        "current_rows":      len(curr),
        "dataset_drift":     dataset_drift,
        "drifted_features":  sorted(set(drifted_features)),
        "feature_drift_scores": feature_drift_scores,
        "html_report":       str(html_path),
    }

    json_path = OUTPUT_DIR / "drift_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    status = "🔴 DRIFT DETECTADO" if dataset_drift else "🟢 Sin drift significativo"
    print(f"{status}")
    print(f"   Features con drift: {drifted_features or 'ninguna'}")
    print(f"   Reporte HTML: {html_path}")
    return summary


if __name__ == "__main__":
    summary = run_drift_report()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
