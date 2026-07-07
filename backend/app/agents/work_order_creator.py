"""
work_order_creator.py — Nodo del grafo: crea la orden de trabajo en Postgres.

Lee sensor_analysis y economic_impact. La prioridad depende del alert_level.
Persiste en la tabla work_orders — la misma que sirve /api/v2/work-orders —
enlazada al agent_run que la creó (created_by_agent_run_id).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog

from ..infra.db.base import AsyncSessionLocal
from ..infra.db.models import WorkOrder
from ..ml.explain import get_machine_id
from .context import current_run_id
from .state import AMIAState

_logger = structlog.get_logger(__name__)

# alert_level → prioridad del enum wo_priority (critical|high|medium|low)
_PRIORITY_MAP: dict[str, str] = {
    "red":    "critical",
    "yellow": "high",
    "green":  "medium",
}

_SOP_MAP: dict[str, str] = {
    "bearing_wear":       "SOP-PM-03",
    "misalignment":       "SOP-PM-07",
    "electrical_failure": "SOP-EL-02",
    "overheating":        "SOP-TH-01",
    "cavitation":         "SOP-HY-04",
    "normal":             "SOP-PM-01",
}
_DEFAULT_SOP = "SOP-GEN-01"


def make_work_order_creator_node() -> Callable[[AMIAState], Awaitable[dict]]:
    """Factoría del nodo WorkOrderCreator (persiste en la tabla work_orders)."""

    async def work_order_creator_node(state: AMIAState) -> dict:
        """
        Lee:    state["sensor_analysis"], state["economic_impact"]
        Escribe: state["work_order"]
        """
        analysis = state.get("sensor_analysis") or {}
        economic = state.get("economic_impact")

        if "error" in analysis or not analysis:
            return {"work_order": None}

        machine_code = analysis.get("machine_id", "")
        alert_level  = analysis.get("alert_level", "green")
        root_cause   = analysis.get("root_cause") or {}
        failure_mode = root_cause.get("failure_mode", "normal")
        priority     = _PRIORITY_MAP.get(alert_level, "medium")
        sop          = _SOP_MAP.get(failure_mode, _DEFAULT_SOP)
        estimated_cost = round(economic.get("total_loss", 0.0), 2) if economic else 0.0

        machine_uuid = get_machine_id(machine_code)
        if machine_uuid is None:
            _logger.warning(
                "work_order: máquina no registrada en la BD, no se persiste",
                machine=machine_code,
            )
            return {"work_order": None}

        wo = WorkOrder(
            machine_id=machine_uuid,
            title=f"{failure_mode.replace('_', ' ').capitalize()} — {machine_code}",
            description=(
                f"Generada automáticamente por el agente IA. "
                f"Modo de fallo diagnosticado: {failure_mode}. SOP de referencia: {sop}."
            ),
            priority=priority,
            status="open",
            estimated_cost=estimated_cost,
            created_by_agent_run_id=current_run_id.get(),
        )
        async with AsyncSessionLocal() as db:
            db.add(wo)
            await db.commit()
            await db.refresh(wo)

        return {"work_order": {
            "work_order_id":  wo.id,
            "machine_id":     machine_code,
            "failure_mode":   failure_mode,
            "priority":       priority,
            "estimated_cost": estimated_cost,
            "sop_reference":  sop,
            "status":         wo.status,
            "created_at":     wo.created_at.isoformat(),
        }}

    return work_order_creator_node
