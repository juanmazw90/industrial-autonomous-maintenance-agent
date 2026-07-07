"""
synthesizer.py — Nodo generador de la respuesta final.

Recibe la query y el contexto acumulado por los agentes anteriores
(retrieved_docs del DocExpert, sensor_analysis del SensorAnalyst)
y genera una respuesta en lenguaje natural con Claude Sonnet.

Es el único nodo que produce texto hacia el usuario.
"""

import os
from typing import cast

import anthropic
from anthropic.types import MessageParam
from langchain_core.callbacks.manager import adispatch_custom_event

from app.infra.settings import settings

from .context import node_usage
from .state import AMIAState

_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_MODEL = settings.llm_synthesizer_model

_SYSTEM_PROMPT = """Eres un experto en mantenimiento industrial.

Responde usando ÚNICAMENTE la información del contexto proporcionado.
Cita las fuentes con [1], [2], etc. al final de cada afirmación relevante.
Si el contexto no contiene la respuesta, dilo claramente — no inventes.
Responde en el mismo idioma que usa el usuario.
Sé preciso y conciso: prioriza procedimientos paso a paso cuando corresponda."""

_ALERT_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
_ALERT_LABEL = {
    "green":  "RIESGO BAJO",
    "yellow": "RIESGO MODERADO",
    "red":    "RIESGO ALTO — ATENCIÓN INMEDIATA",
}


def _build_docs_context(docs: list[dict]) -> str:
    if not docs:
        return "No se encontró documentación relevante."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "desconocido")
        page = doc.get("page", "")
        page_str = f", página {page}" if page else ""
        parts.append(f"[{i}] Fuente: {source}{page_str}\n{doc['text']}")
    return "\n\n".join(parts)


def _build_sensor_context(analysis: dict) -> str:
    if "error" in analysis:
        hint = analysis.get("hint", "")
        available = analysis.get("available_machines", [])
        lines = [f"Error: {analysis['error']}"]
        if hint:
            lines.append(hint)
        if available:
            lines.append(f"Máquinas disponibles: {', '.join(available)}")
        return "\n".join(lines)

    mid   = analysis.get("machine_id", "?")
    prob  = analysis.get("failure_probability", 0.0)
    level = analysis.get("alert_level", "green")
    ts    = analysis.get("as_of_timestamp", "")
    high  = analysis.get("is_high_risk", False)
    thr   = analysis.get("threshold_used", 0.5)

    emoji = _ALERT_EMOJI.get(level, "⚪")
    label = _ALERT_LABEL.get(level, level)

    lines = [
        f"Máquina: {mid}",
        f"Estado: {emoji} {label}",
        f"Probabilidad de fallo en las próximas 24h: {prob:.1%}",
        f"Supera umbral de alerta ({thr:.0%}): {'Sí' if high else 'No'}",
    ]
    if ts:
        lines.append(f"Basado en lectura de: {ts}")

    root_cause = analysis.get("root_cause")
    if root_cause and "error" not in root_cause:
        mode = root_cause["failure_mode"].replace("_", " ").upper()
        conf = root_cause["confidence"]
        lines.append(f"\n🔍 Causa raíz: {mode}  (confianza: {conf:.0%})")
        top3 = sorted(root_cause["probabilities"].items(), key=lambda x: -x[1])[:3]
        dist = " | ".join(f"{k.replace('_', ' ')}: {v:.0%}" for k, v in top3)
        lines.append(f"   Distribución: {dist}")

    return "\n".join(lines)


_URGENCY_EMOJI: dict[str, str] = {"critical": "🔴", "warning": "🟡", "normal": "🟢"}
_URGENCY_LABEL: dict[str, str] = {
    "critical": "CRÍTICO — intervención urgente",
    "warning":  "MODERADO — planificar mantenimiento",
    "normal":   "NORMAL — dentro de parámetros",
}


def _build_rul_context(rul: dict) -> str:
    if "error" in rul:
        return ""
    hours   = rul.get("hours_remaining", 500)
    frac    = rul.get("degradation_fraction", 0.0)
    urgency = rul.get("urgency_level", "normal")
    emoji   = _URGENCY_EMOJI.get(urgency, "⚪")
    label   = _URGENCY_LABEL.get(urgency, urgency)
    lines = [
        f"\n⏱️  Vida útil restante (RUL): {emoji} {label}",
        f"   Horas restantes estimadas: {hours:.0f}h",
        f"   Degradación acumulada: {frac:.0%}",
    ]
    return "\n".join(lines)


_PRIORITY_ES: dict[str, str] = {
    "critical": "CRÍTICA",
    "high":     "ALTA",
    "medium":   "MEDIA",
    "low":      "BAJA",
}


def _build_economic_context(economic: dict, work_order: dict | None) -> str:
    lines = [
        "\n💰 Impacto económico estimado:",
        f"   Pérdida por hora: ${economic['hourly_loss']:,.2f} {economic['currency']}",
        f"   Pérdida total (24h): ${economic['total_loss']:,.2f} {economic['currency']}",
        f"   OEE efectivo: {economic['oee_effective']:.0%}  (nominal: {economic['oee_baseline']:.0%})",
    ]
    if work_order:
        priority_label = _PRIORITY_ES.get(work_order["priority"], work_order["priority"].upper())
        lines += [
            "\n📋 Orden de trabajo creada:",
            f"   ID: {work_order['work_order_id']} — Prioridad: {priority_label}",
            f"   SOP de referencia: {work_order['sop_reference']}",
            f"   Costo estimado: ${work_order['estimated_cost']:,.2f} {economic['currency']}",
            f"   Estado: {work_order['status'].upper()}",
        ]
    return "\n".join(lines)


def _build_sources(docs: list[dict]) -> list[dict]:
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
    Lee:    state["query"], state["retrieved_docs"], state["sensor_analysis"],
            state["rul_prediction"], state["economic_impact"], state["work_order"],
            state["conversation_history"]
    Escribe: state["final_response"], state["sources"]
    """
    query           = state["query"]
    docs            = state.get("retrieved_docs", [])
    sensor_analysis = state.get("sensor_analysis")
    rul_prediction  = state.get("rul_prediction")
    economic_impact = state.get("economic_impact")
    work_order      = state.get("work_order")
    history         = state.get("conversation_history", [])

    if sensor_analysis:
        context = _build_sensor_context(sensor_analysis)
        if rul_prediction and "error" not in rul_prediction:
            context += _build_rul_context(rul_prediction)
        if economic_impact:
            context += _build_economic_context(economic_impact, work_order)
        user_message = f"Datos del sistema de predicción:\n{context}\n\nPregunta del usuario: {query}"
    elif rul_prediction and "error" not in rul_prediction:
        context = _build_rul_context(rul_prediction).lstrip("\n")
        user_message = f"Datos del sistema de predicción:\n{context}\n\nPregunta del usuario: {query}"
    else:
        context = _build_docs_context(docs)
        user_message = f"Contexto:\n{context}\n\nPregunta: {query}"

    messages = cast(list[MessageParam], [
        *history,
        {"role": "user", "content": user_message},
    ])

    # Streaming real: cada delta se emite como custom event para que el endpoint
    # SSE (astream_events → on_custom_event) lo reenvíe al cliente token a token.
    parts: list[str] = []
    emit_ok = True
    async with _client.messages.stream(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            parts.append(text)
            if emit_ok:
                try:
                    await adispatch_custom_event("synthesizer_token", {"content": text})
                except RuntimeError:
                    # Sin contexto runnable (p.ej. nodo invocado directo en tests)
                    emit_ok = False
        final = await stream.get_final_message()

    node_usage.set({
        "model": _MODEL,
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    })

    return {
        "final_response": "".join(parts),
        "sources": _build_sources(docs),
    }
