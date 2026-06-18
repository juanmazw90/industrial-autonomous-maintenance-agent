"""
supervisor.py — Nodo router del grafo multi-agente.

El Supervisor recibe la query y decide a qué agente especialista
derivar el trabajo. Usa Claude con structured output (tool_use)
para garantizar una decisión parseable, no texto libre.

Para V1 solo existen "doc_expert" y "synthesizer".
Los demás agentes se añaden en V2/V3 sin modificar este archivo —
solo se agrega la opción al Literal y al prompt.
"""

import json
import os

import anthropic
from pydantic import BaseModel

from .state import AMIAState

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Schema de la decisión — lo que Claude debe devolver
class RoutingDecision(BaseModel):
    next_agent: str          # "doc_expert" | "synthesizer" (V1)
    reasoning: str           # explicación breve — útil para debugging

# Definición de la tool que fuerza el structured output en Claude
_ROUTING_TOOL = {
    "name": "route_to_agent",
    "description": "Decide qué agente especialista debe manejar la consulta.",
    "input_schema": {
        "type": "object",
        "properties": {
            "next_agent": {
                "type": "string",
                "enum": ["doc_expert", "synthesizer"],
                "description": (
                    "doc_expert: la consulta requiere buscar en manuales o SOPs. "
                    "synthesizer: la consulta puede responderse directamente sin documentación."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Explicación breve de la decisión (1-2 frases).",
            },
        },
        "required": ["next_agent", "reasoning"],
    },
}

_SUPERVISOR_PROMPT = """Eres el supervisor de un sistema de mantenimiento industrial.
Tu única tarea es clasificar la consulta del usuario y decidir qué agente debe actuar.

Agentes disponibles:
- doc_expert: para consultas sobre procedimientos, manuales, SOPs, especificaciones técnicas, \
instrucciones de mantenimiento o reparación.
- synthesizer: para saludos, preguntas generales que no requieren documentación, \
o cuando ya hay suficiente información en el historial de conversación.

Usa siempre la herramienta 'route_to_agent' para registrar tu decisión."""


async def supervisor_node(state: AMIAState) -> dict:
    """
    Lee:    state["query"], state["conversation_history"]
    Escribe: state["next_agent"]
    """
    messages = [
        *state.get("conversation_history", []),
        {"role": "user", "content": state["query"]},
    ]

    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",   # Haiku para routing — rápido y barato
        max_tokens=256,
        system=_SUPERVISOR_PROMPT,
        tools=[_ROUTING_TOOL],
        tool_choice={"type": "any"},          # fuerza el uso de la tool
        messages=messages,
    )

    # Extraer la decisión del tool_use block
    for block in response.content:
        if block.type == "tool_use" and block.name == "route_to_agent":
            decision = RoutingDecision(**block.input)
            return {"next_agent": decision.next_agent}

    # Fallback seguro si Claude no usa la tool (no debería ocurrir con tool_choice="any")
    return {"next_agent": "doc_expert"}
