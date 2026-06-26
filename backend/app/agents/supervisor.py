"""
supervisor.py — Nodo router del grafo multi-agente.

El Supervisor recibe la query y decide a qué agente especialista
derivar el trabajo. Usa Claude con structured output (tool_use)
para garantizar una decisión parseable, no texto libre.

Agentes disponibles:
- doc_expert:      consultas sobre manuales, SOPs y documentación técnica
- sensor_analyst:  consultas sobre estado, riesgo o predicción de fallos de una máquina
- synthesizer:     respuestas directas sin necesidad de herramientas adicionales
"""

import os

import anthropic
from pydantic import BaseModel

from .state import AMIAState

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


class RoutingDecision(BaseModel):
    next_agent: str
    reasoning: str


_ROUTING_TOOL = {
    "name": "route_to_agent",
    "description": "Decide qué agente especialista debe manejar la consulta.",
    "input_schema": {
        "type": "object",
        "properties": {
            "next_agent": {
                "type": "string",
                "enum": ["doc_expert", "sensor_analyst", "rul_analyst", "synthesizer"],
                "description": (
                    "doc_expert: la consulta requiere buscar en manuales, SOPs o especificaciones técnicas. "
                    "sensor_analyst: la consulta pregunta por el estado, riesgo de fallo o salud de una "
                    "máquina específica (contiene un ID como COMP-001, MOTOR-001, PUMP-002, etc.). "
                    "rul_analyst: la consulta pregunta cuánto tiempo le queda a una máquina, su vida útil "
                    "restante o nivel de degradación acumulada. "
                    "synthesizer: saludos, preguntas generales o cuando el historial ya tiene la respuesta."
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

_SUPERVISOR_PROMPT = """Eres el supervisor de un sistema de mantenimiento industrial con IA.
Tu única tarea es clasificar la consulta del usuario y decidir qué agente debe actuar.

Agentes disponibles:
- doc_expert: para consultas sobre procedimientos, manuales, SOPs, especificaciones técnicas
  o instrucciones de mantenimiento y reparación.
- sensor_analyst: para consultas sobre el estado actual, riesgo de fallo, salud o predicción
  de una máquina específica. Se activa cuando la consulta contiene un ID de máquina
  (COMP-001, MOTOR-001, MOTOR-002, PUMP-001, PUMP-002).
- rul_analyst: para consultas sobre cuánto tiempo le queda a una máquina antes de fallar,
  su vida útil restante en horas o su nivel de degradación acumulada.
- synthesizer: para saludos, preguntas generales que no requieren documentación ni datos
  de sensores, o cuando el historial ya contiene suficiente información.

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
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_SUPERVISOR_PROMPT,
        tools=[_ROUTING_TOOL],
        tool_choice={"type": "any"},
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "route_to_agent":
            decision = RoutingDecision(**block.input)
            return {"next_agent": decision.next_agent}

    return {"next_agent": "doc_expert"}
