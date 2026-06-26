"""
rul_analyst.py — Nodo del grafo: predice la vida útil restante (RUL) de la máquina.

Lee machine_id desde sensor_analysis (cadena de alertas) o desde la query
del usuario (ruta directa desde supervisor). Escribe rul_prediction en estado.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .state import AMIAState

_MACHINE_ID_RE = re.compile(r"\b((?:COMP|MOTOR|PUMP)-\d{3})\b", re.IGNORECASE)


def make_rul_analyst_node(
    predict_rul_fn: Callable[[str], dict[str, Any]],
) -> Callable[[AMIAState], dict]:

    def rul_analyst_node(state: AMIAState) -> dict:
        """
        Lee:    state["sensor_analysis"] (machine_id upstream) o state["query"]
        Escribe: state["rul_prediction"]
        """
        # Preferir machine_id del sensor_analyst previo (cadena de alertas)
        sensor_analysis = state.get("sensor_analysis") or {}
        machine_id = sensor_analysis.get("machine_id")

        if not machine_id:
            query = state["query"]
            match = _MACHINE_ID_RE.search(query.upper())
            if not match:
                return {
                    "rul_prediction": {
                        "error": "No se encontró un ID de máquina en la consulta.",
                        "hint": "Menciona el ID de la máquina (ej. COMP-001, MOTOR-001, PUMP-002).",
                        "available_machines": ["COMP-001", "MOTOR-001", "MOTOR-002", "PUMP-001", "PUMP-002"],
                    }
                }
            machine_id = match.group(1)

        result = predict_rul_fn(machine_id)
        return {"rul_prediction": result}

    return rul_analyst_node
