"""
graph.py — Ensamblado del grafo multi-agente con LangGraph.

Flujo V2:
  supervisor → doc_expert    → synthesizer → END   (consultas de documentación)
  supervisor → sensor_analyst → synthesizer → END  (consultas de predicción de fallos)
  supervisor → synthesizer → END                   (respuestas directas)

Para añadir un agente nuevo (V3+):
  1. Importar el nodo
  2. graph.add_node(...)
  3. Añadir la arista specialist → synthesizer
  4. Añadir la opción en el dict de conditional_edges
  5. Actualizar el enum en supervisor.py
"""

from typing import Any

from langgraph.graph import END, StateGraph

from .agents.doc_expert import make_doc_expert_node
from .agents.economic_analyst import make_economic_analyst_node
from .agents.sensor_analyst import make_sensor_analyst_node
from .agents.state import AMIAState
from .agents.supervisor import supervisor_node
from .agents.synthesizer import synthesizer_node
from .agents.work_order_creator import make_work_order_creator_node
from .services.cmms import CMMS
from .services.predictor import FailurePredictor
from .services.rag_config import RAGConfig
from .services.rca_predictor import RCAPredictor
from .services.retrieval import Retriever
from .services.tools import build_predict_failure_tool, build_predict_rca_tool, build_search_tool


def _route(state: AMIAState) -> str:
    return state.get("next_agent", "doc_expert")


def _route_after_sensor(state: AMIAState) -> str:
    """
    Routing condicional después de sensor_analyst.
    Si hay riesgo real (yellow/red) y root_cause válido → economic_analyst.
    En cualquier otro caso → synthesizer directamente.
    """
    analysis = state.get("sensor_analysis") or {}
    if "error" in analysis or not analysis:
        return "synthesizer"
    alert     = analysis.get("alert_level", "green")
    root_cause = analysis.get("root_cause")
    if alert == "green" or not root_cause or "error" in root_cause:
        return "synthesizer"
    return "economic_analyst"


def build_graph(
    config: RAGConfig,
    predictor: FailurePredictor | None = None,
    retriever: Retriever | None = None,
    rca_predictor: RCAPredictor | None = None,
    cmms: CMMS | None = None,
) -> Any:
    """
    Construye y compila el grafo. Se llama una vez al arrancar la app.

    Args:
        config:    configuración RAG (embeddings, Qdrant, etc.)
        predictor: instancia de FailurePredictor (puede estar sin inicializar todavía)
        retriever: instancia de Retriever; si None se crea internamente.
                   Pasar explícitamente permite compartir el embedder con SemanticCache.
    """
    if retriever is None:
        retriever = Retriever(config)
    search_fn  = build_search_tool(retriever)

    graph = StateGraph(AMIAState)

    # ── Nodos base ────────────────────────────────────────────────────────────
    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("doc_expert",  make_doc_expert_node(search_fn))
    graph.add_node("synthesizer", synthesizer_node)

    if predictor is not None:
        predict_fn     = build_predict_failure_tool(predictor)
        predict_rca_fn = build_predict_rca_tool(rca_predictor) if rca_predictor is not None else None
        graph.add_node("sensor_analyst", make_sensor_analyst_node(predict_fn, predict_rca_fn))

    # ── Nodos V3 (siempre registrados; solo alcanzables desde sensor_analyst) ─
    if cmms is None:
        cmms = CMMS()
    graph.add_node("economic_analyst",   make_economic_analyst_node())
    graph.add_node("work_order_creator", make_work_order_creator_node(cmms))

    # ── Aristas ───────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    routing_map: dict[str, str] = {
        "doc_expert":  "doc_expert",
        "synthesizer": "synthesizer",
    }
    if predictor is not None:
        routing_map["sensor_analyst"] = "sensor_analyst"

    graph.add_conditional_edges("supervisor", _route, routing_map)

    graph.add_edge("doc_expert", "synthesizer")

    if predictor is not None:
        # Routing condicional: verde → synthesizer, amarillo/rojo → economic_analyst
        graph.add_conditional_edges(
            "sensor_analyst",
            _route_after_sensor,
            {"economic_analyst": "economic_analyst", "synthesizer": "synthesizer"},
        )

    # Cadena V3
    graph.add_edge("economic_analyst",   "work_order_creator")
    graph.add_edge("work_order_creator", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()


_graph_instance = None


def get_graph(
    config: RAGConfig | None = None,
    predictor: FailurePredictor | None = None,
    retriever: Retriever | None = None,
    rca_predictor: RCAPredictor | None = None,
    cmms: CMMS | None = None,
) -> Any:
    """Devuelve la instancia compilada del grafo (singleton)."""
    global _graph_instance
    if _graph_instance is None:
        if config is None:
            config = RAGConfig()
        _graph_instance = build_graph(config, predictor, retriever, rca_predictor, cmms)
    return _graph_instance
