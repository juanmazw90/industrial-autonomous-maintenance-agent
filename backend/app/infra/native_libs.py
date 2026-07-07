"""Workaround del conflicto libomp entre XGBoost y torch (macOS).

Ambas librerías empaquetan su propio runtime OpenMP y el orden de
inicialización importa: si torch carga primero, cualquier operación nativa
de XGBoost segfaultea el proceso (SIGSEGV, exit 139). Inicializar XGBoost
primero deja ambos runtimes operativos.

ensure_openmp_order() debe ejecutarse ANTES de importar torch /
sentence-transformers (main.py lo llama justo después de cargar .env).
"""
from __future__ import annotations


def ensure_openmp_order() -> None:
    import numpy as np
    import xgboost as xgb

    warmup = xgb.XGBClassifier(n_estimators=1)
    warmup.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
