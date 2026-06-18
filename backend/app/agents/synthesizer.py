"""
synthesizer.py — Nodo generador de la respuesta final.

Recibe la query y los docs recuperados por DocExpert,
y genera una respuesta citando fuentes con Claude Sonnet.

Es el único nodo que produce texto hacia el usuario.
"""

import os

import anthropic

from .state import AMIAState

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

_SYSTEM_PROMPT = """Eres un experto en mantenimiento industrial.

Responde usando ÚNICAMENTE la información del contexto proporcionado.
Cita las fuentes con [1], [2], etc. al final de cada afirmación relevante.
Si el contexto no contiene la respuesta, dilo claramente — no inventes.
Responde en el mismo idioma que usa el usuario.
Sé preciso y conciso: prioriza procedimientos paso a paso cuando corresponda."""


def _build_context(docs: list[dict]) -> str:
    """Formatea los chunks recuperados como contexto numerado para el prompt."""
    if not docs:
        return "No se encontró documentación relevante."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "desconocido")
        page = doc.get("page", "")
        page_str = f", página {page}" if page else ""
        parts.append(f"[{i}] Fuente: {source}{page_str}\n{doc['text']}")
    return "\n\n".join(parts)


def _build_sources(docs: list[dict]) -> list[dict]:
    """Construye la lista de fuentes para devolver al frontend."""
    return [
        {
            "index": i + 1,
            "source": doc.get("source", "desconocido"),
            "page": doc.get("page", ""),
            "rerank_score": doc.get("rerank_score", 0.0),
        }
        for i, doc in enumerate(docs)
    ]


async def synthesizer_node(state: AMIAState) -> dict:
    """
    Lee:    state["query"], state["retrieved_docs"], state["conversation_history"]
    Escribe: state["final_response"], state["sources"]
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    history = state.get("conversation_history", [])

    context = _build_context(docs)

    user_message = f"Contexto:\n{context}\n\nPregunta: {query}"

    messages = [
        *history,
        {"role": "user", "content": user_message},
    ]

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text if response.content else ""

    return {
        "final_response": answer,
        "sources": _build_sources(docs),
    }
