"""
graph.py — Ensamblado del grafo multi-agente con LangGraph.

El grafo define el flujo de ejecución. Para añadir un agente nuevo (V2, V3):
  1. Importar el nodo
  2. graph.add_node(...)
  3. Añadir aristas desde/hacia ese nodo
  4. Agregar la opción en _route()
No hay que tocar los agentes existentes.
"""

from typing import Any

from langgraph.graph import END, StateGraph

from .agents.doc_expert import make_doc_expert_node
from .agents.state import AMIAState
from .agents.supervisor import supervisor_node
from .agents.synthesizer import synthesizer_node
from .services.rag_config import RAGConfig
from .services.retrieval import Retriever
from .services.tools import build_search_tool


def _route(state: AMIAState) -> str:
    """
    Función de routing condicional.
    Lee la decisión del Supervisor y devuelve el nombre del siguiente nodo.
    """
    return state.get("next_agent", "doc_expert")


def build_graph(config: RAGConfig) -> Any:
    """
    Construye y compila el grafo. Se llama una vez al arrancar la app.

    Recibe RAGConfig para instanciar el Retriever y las tools,
    que luego se inyectan en los nodos via el patrón factory.
    """
    # Instanciar dependencias (una sola vez, no en cada request)
    retriever = Retriever(config)
    search_fn = build_search_tool(retriever)

    # Crear el grafo con el esquema de estado
    graph = StateGraph(AMIAState)

    # ── Nodos ────────────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("doc_expert", make_doc_expert_node(search_fn))  # factory pattern
    graph.add_node("synthesizer", synthesizer_node)

    # Nodos futuros (descomentar en V2/V3):
    # graph.add_node("sensor_analyst", make_sensor_analyst_node(predictor))
    # graph.add_node("work_order",     make_work_order_node(cmms_client))

    # ── Aristas ──────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # El supervisor decide condicionalmente a dónde ir
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {
            "doc_expert":  "doc_expert",
            "synthesizer": "synthesizer",   # respuesta directa sin RAG
        },
    )

    # Todos los especialistas convergen en el synthesizer
    graph.add_edge("doc_expert", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Singleton del grafo — se importa desde main.py
_graph_instance = None


def get_graph(config: RAGConfig | None = None):
    """Devuelve la instancia compilada del grafo (singleton)."""
    global _graph_instance
    if _graph_instance is None:
        if config is None:
            config = RAGConfig()
        _graph_instance = build_graph(config)
    return _graph_instance
