"""
sensor_analyst.py — Agente especialista en análisis de sensores y predicción de fallos.

Responsabilidad única: dado que el Supervisor detectó una consulta sobre el estado
de una máquina, extrae el machine_id de la query e invoca predict_failure_risk.

El resultado se almacena en state["sensor_analysis"] para que el Synthesizer
lo formatee como respuesta al usuario.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .state import AMIAState

# Patrón para IDs de máquina: COMP-001, MOTOR-002, PUMP-001, etc.
_MACHINE_ID_RE = re.compile(r"\b([A-Z]+-\d+)\b")


def make_sensor_analyst_node(
    predict_fn: Callable[[str], dict[str, Any]],
    predict_rca_fn: Callable[[str], dict[str, Any]] | None = None,
) -> Callable[[AMIAState], dict]:
    """
    Factoría del nodo SensorAnalyst.

    Args:
        predict_fn:     función síncrona → {failure_probability, alert_level, ...}
        predict_rca_fn: función síncrona opcional → {failure_mode, confidence, ...}
                        Si está disponible, se llama automáticamente cuando
                        alert_level != "green" y enriquece el resultado con "root_cause".
    """

    def sensor_analyst_node(state: AMIAState) -> dict:
        """
        Lee:    state["query"]
        Escribe: state["sensor_analysis"]
        """
        query = state["query"]

        # Buscar machine_id en la query (case-insensitive)
        match = _MACHINE_ID_RE.search(query.upper())
        if not match:
            return {
                "sensor_analysis": {
                    "error": "No se encontró un ID de máquina en la consulta.",
                    "hint": "Menciona el ID de la máquina (ej. COMP-001, MOTOR-001, PUMP-002).",
                    "available_machines": ["COMP-001", "MOTOR-001", "MOTOR-002", "PUMP-001", "PUMP-002"],
                }
            }

        machine_id = match.group(1)
        result = predict_fn(machine_id)

        # RCA: diagnóstico automático cuando hay riesgo detectado
        if predict_rca_fn is not None and result.get("alert_level") != "green":
            rca = predict_rca_fn(machine_id)
            if "error" not in rca:
                result["root_cause"] = rca

        return {"sensor_analysis": result}

    return sensor_analyst_node
