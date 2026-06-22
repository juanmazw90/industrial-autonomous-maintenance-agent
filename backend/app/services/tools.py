"""
tools.py — Funciones que los agentes pueden invocar como herramientas.

Cada tool recibe argumentos simples y devuelve un dict serializable,
para que el resultado pueda almacenarse en AMIAState.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amia_shared.schemas import MACHINE_CONFIGS
from .retrieval import Retriever, RetrievedChunk

if TYPE_CHECKING:
    from .predictor import FailurePredictor


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
            return predictor.predict(machine_id)
        except ValueError as e:
            return {"error": str(e)}

    return predict_failure_risk
