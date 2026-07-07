"""
feature_store.py — Carga única del histórico + features para los 3 predictores.

Antes cada predictor (failure/RCA/RUL) releía el parquet completo y reconstruía
las features en su initialize() — 3× el mismo trabajo (~30s cada uno) en el boot.
El lifespan computa un FeatureBundle una vez y lo inyecta en los tres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from amia_shared.features import build_features, compute_baselines


@dataclass
class FeatureBundle:
    raw: pd.DataFrame
    features: pd.DataFrame
    baselines: dict[str, dict[str, float]]


def load_feature_bundle(data_path: Path) -> FeatureBundle:
    df = pd.read_parquet(data_path)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return FeatureBundle(raw=df, features=build_features(df), baselines=compute_baselines(df))
