"""
economic_analyst.py — Nodo del grafo: calcula impacto económico del riesgo detectado.

Se activa solo cuando sensor_analysis indica alert_level != "green" y existe root_cause válido.
"""
from __future__ import annotations

from collections.abc import Callable

from amia_shared.schemas import MACHINE_CONFIGS

from .state import AMIAState
from ..services.economic_impact import calculate_loss


def make_economic_analyst_node() -> Callable[[AMIAState], dict]:
    """Factoría del nodo EconomicAnalyst. Sin dependencias inyectadas."""

    def economic_analyst_node(state: AMIAState) -> dict:
        """
        Lee:    state["sensor_analysis"]
        Escribe: state["economic_impact"]
        """
        analysis = state.get("sensor_analysis") or {}

        if "error" in analysis or not analysis:
            return {"economic_impact": None}

        alert_level = analysis.get("alert_level", "green")
        if alert_level == "green":
            return {"economic_impact": None}

        root_cause = analysis.get("root_cause")
        if not root_cause or "error" in root_cause:
            return {"economic_impact": None}

        machine_id  = analysis.get("machine_id", "")
        machine_cfg = MACHINE_CONFIGS.get(machine_id, {})
        machine_type = machine_cfg.get("type", "centrifugal_pump")

        impact = calculate_loss(
            machine_id=machine_id,
            machine_type=machine_type,
            alert_level=alert_level,
            hours_at_risk=24.0,
        )

        return {"economic_impact": impact}

    return economic_analyst_node
