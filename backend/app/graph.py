"""
graph.py — Ensamblado del grafo multi-agente con LangGraph.

Flujo V4:
  supervisor → doc_expert    → synthesizer → END          (documentación)
  supervisor → sensor_analyst → [verde]    → synthesizer  (máquina sana)
  supervisor → sensor_analyst → [alerta]   → rul_analyst → [sin impacto] → synthesizer
                                            → [con impacto] → economic_analyst → work_order_creator → synthesizer
  supervisor → rul_analyst   → synthesizer → END          (consulta directa de RUL)
  supervisor → synthesizer   → END                        (respuesta directa)
"""

from collections.abc import Hashable
from typing import Any

from langgraph.graph import END, StateGraph

from .agents.doc_expert import make_doc_expert_node
from .agents.economic_analyst import make_economic_analyst_node
from .agents.instrumentation import instrument_async_tool, instrument_node, instrument_tool
from .agents.rul_analyst import make_rul_analyst_node
from .agents.sensor_analyst import make_sensor_analyst_node
from .agents.state import AMIAState
from .agents.supervisor import supervisor_node
from .agents.synthesizer import synthesizer_node
from .agents.work_order_creator import make_work_order_creator_node
from .rag.metrics import InstrumentedRetriever
from .services.predictor import FailurePredictor
from .services.rag_config import RAGConfig
from .services.rca_predictor import RCAPredictor
from .services.retrieval import Retriever
from .services.rul_predictor import RULPredictor
from .services.tools import (
    build_predict_failure_tool,
    build_predict_rca_tool,
    build_predict_rul_tool,
    build_search_tool,
)


def _route(state: AMIAState) -> str:
    return state.get("next_agent", "doc_expert")


def _route_after_sensor(state: AMIAState) -> str:
    """
    Routing condicional después de sensor_analyst.
    Amarillo/rojo + root_cause válido → rul_analyst (V4).
    Cualquier otro caso → synthesizer.
    """
    analysis = state.get("sensor_analysis") or {}
    if "error" in analysis or not analysis:
        return "synthesizer"
    alert      = analysis.get("alert_level", "green")
    root_cause = analysis.get("root_cause")
    if alert == "green" or not root_cause or "error" in root_cause:
        return "synthesizer"
    return "rul_analyst"


def _route_after_rul(state: AMIAState) -> str:
    """
    Routing condicional después de rul_analyst.
    Si venimos de la cadena de alertas (sensor_analysis con amarillo/rojo) → economic_analyst.
    En consulta directa de RUL (sin sensor_analysis) → synthesizer.
    """
    analysis = state.get("sensor_analysis") or {}
    if "error" in analysis or not analysis:
        return "synthesizer"
    alert      = analysis.get("alert_level", "green")
    root_cause = analysis.get("root_cause")
    if alert == "green" or not root_cause or "error" in root_cause:
        return "synthesizer"
    return "economic_analyst"


def build_graph(
    config: RAGConfig,
    predictor: FailurePredictor | None = None,
    retriever: "Retriever | InstrumentedRetriever | None" = None,
    rca_predictor: RCAPredictor | None = None,
    rul_predictor: RULPredictor | None = None,
) -> Any:
    """Construye y compila el grafo. Se llama una vez al arrancar la app."""
    if retriever is None:
        retriever = Retriever(config)
    search_fn = instrument_async_tool("search_documentation")(build_search_tool(retriever))

    graph = StateGraph(AMIAState)

    # ── Nodos base ────────────────────────────────────────────────────────────
    graph.add_node("supervisor",  instrument_node("supervisor")(supervisor_node))
    graph.add_node("doc_expert",  instrument_node("doc_expert")(make_doc_expert_node(search_fn)))
    graph.add_node("synthesizer", instrument_node("synthesizer")(synthesizer_node))

    if predictor is not None:
        predict_fn = instrument_tool("predict_failure")(build_predict_failure_tool(predictor))
        predict_rca_fn = (
            instrument_tool("predict_rca")(build_predict_rca_tool(rca_predictor))
            if rca_predictor is not None
            else None
        )
        graph.add_node("sensor_analyst", instrument_node("sensor_analyst")(
            make_sensor_analyst_node(predict_fn, predict_rca_fn)
        ))

    # ── Nodos V3/V4 (siempre registrados) ────────────────────────────────────
    graph.add_node("economic_analyst",   instrument_node("economic_analyst")(make_economic_analyst_node()))
    graph.add_node("work_order_creator", instrument_node("work_order_creator")(make_work_order_creator_node()))

    if rul_predictor is None:
        rul_predictor = RULPredictor()
    predict_rul_fn = instrument_tool("predict_rul")(build_predict_rul_tool(rul_predictor))
    graph.add_node("rul_analyst", instrument_node("rul_analyst")(make_rul_analyst_node(predict_rul_fn)))

    # ── Aristas ───────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    routing_map: dict[Hashable, str] = {
        "doc_expert":  "doc_expert",
        "rul_analyst": "rul_analyst",
        "synthesizer": "synthesizer",
    }
    if predictor is not None:
        routing_map["sensor_analyst"] = "sensor_analyst"

    graph.add_conditional_edges("supervisor", _route, routing_map)

    graph.add_edge("doc_expert", "synthesizer")

    if predictor is not None:
        # Verde → synthesizer; amarillo/rojo → rul_analyst
        graph.add_conditional_edges(
            "sensor_analyst",
            _route_after_sensor,
            {"rul_analyst": "rul_analyst", "synthesizer": "synthesizer"},
        )

    # Routing desde rul_analyst: consulta directa → synthesizer; cadena de alertas → economic_analyst
    graph.add_conditional_edges(
        "rul_analyst",
        _route_after_rul,
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
    retriever: "Retriever | InstrumentedRetriever | None" = None,
    rca_predictor: RCAPredictor | None = None,
    rul_predictor: RULPredictor | None = None,
) -> Any:
    """Devuelve la instancia compilada del grafo (singleton)."""
    global _graph_instance
    if _graph_instance is None:
        if config is None:
            config = RAGConfig()
        _graph_instance = build_graph(config, predictor, retriever, rca_predictor, rul_predictor)
    return _graph_instance
