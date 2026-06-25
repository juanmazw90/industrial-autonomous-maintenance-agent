"""
work_order_creator.py — Nodo del grafo: crea orden de trabajo en el CMMS mock.

Lee sensor_analysis y economic_impact. La prioridad depende del alert_level.
"""
from __future__ import annotations

from collections.abc import Callable

from .state import AMIAState
from ..services.cmms import CMMS

_PRIORITY_MAP: dict[str, str] = {
    "red":    "urgent",
    "yellow": "high",
    "green":  "normal",
}


def make_work_order_creator_node(cmms: CMMS) -> Callable[[AMIAState], dict]:
    """
    Factoría del nodo WorkOrderCreator.

    Args:
        cmms: instancia del CMMS mock (inyectada desde graph.py)
    """

    def work_order_creator_node(state: AMIAState) -> dict:
        """
        Lee:    state["sensor_analysis"], state["economic_impact"]
        Escribe: state["work_order"]
        """
        analysis = state.get("sensor_analysis") or {}
        economic  = state.get("economic_impact")

        if "error" in analysis or not analysis:
            return {"work_order": None}

        machine_id   = analysis.get("machine_id", "")
        alert_level  = analysis.get("alert_level", "green")
        root_cause   = analysis.get("root_cause") or {}
        failure_mode = root_cause.get("failure_mode", "normal")
        priority     = _PRIORITY_MAP.get(alert_level, "normal")

        order = cmms.create_work_order(
            machine_id=machine_id,
            failure_mode=failure_mode,
            priority=priority,
            economic_data=economic,
        )

        return {"work_order": order}

    return work_order_creator_node
