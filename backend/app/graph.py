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
from .agents.sensor_analyst import make_sensor_analyst_node
from .agents.state import AMIAState
from .agents.supervisor import supervisor_node
from .agents.synthesizer import synthesizer_node
from .services.predictor import FailurePredictor
from .services.rag_config import RAGConfig
from .services.retrieval import Retriever
from .services.tools import build_predict_failure_tool, build_search_tool


def _route(state: AMIAState) -> str:
    return state.get("next_agent", "doc_expert")


def build_graph(config: RAGConfig, predictor: FailurePredictor | None = None) -> Any:
    """
    Construye y compila el grafo. Se llama una vez al arrancar la app.

    Args:
        config:    configuración RAG (embeddings, Qdrant, etc.)
        predictor: instancia de FailurePredictor (puede estar sin inicializar todavía —
                   se inicializa en el lifespan handler tras arrancar el grafo)
    """
    retriever  = Retriever(config)
    search_fn  = build_search_tool(retriever)

    graph = StateGraph(AMIAState)

    # ── Nodos ─────────────────────────────────────────────────────────────────
    graph.add_node("supervisor",     supervisor_node)
    graph.add_node("doc_expert",     make_doc_expert_node(search_fn))
    graph.add_node("synthesizer",    synthesizer_node)

    if predictor is not None:
        predict_fn = build_predict_failure_tool(predictor)
        graph.add_node("sensor_analyst", make_sensor_analyst_node(predict_fn))
    else:
        # Sin predictor el nodo no existe — el supervisor nunca lo elegirá
        # porque sensor_analyst no estará en el enum de routing
        pass

    # Nodos futuros (V3):
    # graph.add_node("work_order", make_work_order_node(cmms_client))

    # ── Aristas ───────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    routing_map: dict[str, str] = {
        "doc_expert":  "doc_expert",
        "synthesizer": "synthesizer",
    }
    if predictor is not None:
        routing_map["sensor_analyst"] = "sensor_analyst"

    graph.add_conditional_edges("supervisor", _route, routing_map)

    graph.add_edge("doc_expert",     "synthesizer")
    if predictor is not None:
        graph.add_edge("sensor_analyst", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


_graph_instance = None


def get_graph(config: RAGConfig | None = None, predictor: FailurePredictor | None = None) -> Any:
    """Devuelve la instancia compilada del grafo (singleton)."""
    global _graph_instance
    if _graph_instance is None:
        if config is None:
            config = RAGConfig()
        _graph_instance = build_graph(config, predictor)
    return _graph_instance
