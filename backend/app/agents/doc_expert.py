"""
doc_expert.py — Agente especialista en recuperación de documentación.

Responsabilidad única: dada una query, busca en Qdrant y devuelve
los chunks más relevantes. No genera texto — solo recupera.

Patrón factory: `make_doc_expert_node(search_fn)` devuelve el nodo
con las dependencias ya inyectadas (el Retriever no se puede pasar
como argumento directo a un nodo de LangGraph).
"""

from collections.abc import Callable, Coroutine
from typing import Any

from .state import AMIAState


def make_doc_expert_node(
    search_documentation: Callable[..., Coroutine[Any, Any, list[dict]]],
) -> Callable[[AMIAState], Coroutine[Any, Any, dict]]:
    """
    Factoría del nodo DocExpert.

    Args:
        search_documentation: función async que recibe (query, top_k)
                               y devuelve lista de chunks desde Qdrant.
    Returns:
        Función nodo compatible con LangGraph.
    """

    async def doc_expert_node(state: AMIAState) -> dict:
        """
        Lee:   state["query"]
        Escribe: state["retrieved_docs"]
        """
        query = state["query"]
        docs = await search_documentation(query, top_k=3)
        return {"retrieved_docs": docs}

    return doc_expert_node
