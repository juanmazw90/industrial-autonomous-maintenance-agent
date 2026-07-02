"""
tools.py — Funciones que los agentes pueden invocar como herramientas.

Cada tool recibe argumentos simples y devuelve un dict serializable,
para que el resultado pueda almacenarse en AMIAState.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from amia_shared.schemas import MACHINE_CONFIGS
from app.agents.context import get_event_loop
from app.ml.explain import save_prediction
from .retrieval import Retriever, RetrievedChunk

if TYPE_CHECKING:
    from .predictor import FailurePredictor
    from .rca_predictor import RCAPredictor
    from .rul_predictor import RULPredictor


def _fire_save_prediction(**kwargs) -> None:
    """Schedule save_prediction coroutine from a thread-pool context."""
    loop = get_event_loop()
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(save_prediction(**kwargs), loop)


def build_search_tool(retriever: Retriever):
    """
    Factoría: devuelve una función de búsqueda con el retriever ya inyectado.
    Se llama una vez al arrancar la app, no en cada request.
    """
    async def search_documentation(query: str, top_k: int = 3) -> list[dict]:
        """Busca en la base de conocimiento (manuales y SOPs) y devuelve chunks relevantes."""
        chunks: list[RetrievedChunk] = await retriever.retrieve(query, top_k=top_k)
        return [
            {
                "text": c.text,
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", ""),
                "vector_score": round(c.vector_score, 4),
                "rerank_score": round(c.rerank_score, 4),
            }
            for c in chunks
        ]

    return search_documentation


def get_machine_info(machine_id: str) -> dict:
    """Devuelve la configuración nominal de una máquina por su ID."""
    cfg = MACHINE_CONFIGS.get(machine_id.upper())
    if not cfg:
        available = list(MACHINE_CONFIGS.keys())
        return {"error": f"Máquina '{machine_id}' no encontrada.", "available": available}
    return {"machine_id": machine_id.upper(), **cfg}


def build_predict_failure_tool(predictor: FailurePredictor):
    """
    Factoría: devuelve la tool de predicción de fallos con el predictor ya inyectado.
    El Supervisor la invoca cuando detecta un machine_id en la query del usuario.
    """
    def predict_failure_risk(machine_id: str) -> dict:
        """
        Predice el riesgo de fallo de una máquina en las próximas 24 horas.

        Args:
            machine_id: ID de la máquina (ej. 'COMP-001', 'PUMP-002', 'MOTOR-001')

        Returns:
            Dict con failure_probability, risk_score, alert_level (green/yellow/red),
            is_high_risk y el timestamp de la última lectura usada.
        """
        if not predictor.initialized:
            return {
                "error": "El predictor no está disponible.",
                "hint": "Ejecuta train_failure_prediction.py para entrenar el modelo.",
            }
        try:
            result = predictor.predict(machine_id)
            _fire_save_prediction(
                machine_code=machine_id,
                model_name="amia-failure-model",
                model_version=None,
                prediction=result,
                probability=result.get("failure_probability"),
                shap_model=predictor.model,
                shap_features=predictor._latest_features.get(machine_id.upper()),
                shap_feature_names=predictor.feature_cols,
            )
            return result
        except ValueError as e:
            return {"error": str(e)}

    return predict_failure_risk


def build_predict_rul_tool(predictor: RULPredictor):
    """
    Factoría: devuelve la tool de predicción de RUL con el predictor inyectado.
    """
    def predict_remaining_useful_life(machine_id: str) -> dict:
        """
        Predice las horas de vida útil restante de una máquina.

        Args:
            machine_id: ID de la máquina (ej. 'COMP-001', 'PUMP-002')

        Returns:
            Dict con hours_remaining, degradation_fraction, urgency_level y timestamp.
        """
        if not predictor.initialized:
            return {
                "error": "RULPredictor no disponible.",
                "hint": "Ejecuta train_rul.py para entrenar el modelo.",
            }
        try:
            result = predictor.predict(machine_id)
            _fire_save_prediction(
                machine_code=machine_id,
                model_name="amia-rul-model",
                model_version=None,
                prediction=result,
                probability=result.get("degradation_fraction"),
            )
            return result
        except ValueError as e:
            return {"error": str(e)}

    return predict_remaining_useful_life


def build_predict_rca_tool(predictor: RCAPredictor):
    """
    Factoría: devuelve la tool de diagnóstico de causa raíz con el predictor inyectado.
    Se invoca desde sensor_analyst cuando alert_level != "green".
    """
    def predict_root_cause(machine_id: str) -> dict:
        """
        Diagnostica el modo de fallo activo en una máquina.

        Args:
            machine_id: ID de la máquina (ej. 'COMP-001', 'PUMP-002')

        Returns:
            Dict con failure_mode, confidence, probabilities (por clase) y timestamp.
        """
        if not predictor.initialized:
            return {
                "error": "RCAPredictor no disponible.",
                "hint": "Ejecuta train_rca.py para entrenar el modelo.",
            }
        try:
            return predictor.predict(machine_id)
        except ValueError as e:
            return {"error": str(e)}

    return predict_root_cause
