"""
cmms.py — CMMS mock (Computerized Maintenance Management System).

Simula creación y consulta de órdenes de trabajo en memoria.
En producción se reemplaza por un cliente REST a SAP PM, Maximo, etc.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

_SOP_MAP: dict[str, str] = {
    "bearing_wear":       "SOP-PM-03",
    "misalignment":       "SOP-PM-07",
    "electrical_failure": "SOP-EL-02",
    "overheating":        "SOP-TH-01",
    "cavitation":         "SOP-HY-04",
    "normal":             "SOP-PM-01",
}

_DEFAULT_SOP = "SOP-GEN-01"


class CMMS:
    """CMMS mock thread-safe. El singleton vive en build_graph()."""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._counter: int = 0
        self._lock = threading.Lock()

    def create_work_order(
        self,
        machine_id: str,
        failure_mode: str,
        priority: str,
        economic_data: dict | None = None,
    ) -> dict:
        """
        Crea una orden de trabajo y la almacena en memoria.

        Args:
            machine_id:    ID de la máquina afectada
            failure_mode:  modo de fallo diagnosticado (del RCA)
            priority:      urgent | high | normal
            economic_data: dict de economic_impact (opcional)

        Returns:
            dict con work_order_id, machine_id, failure_mode, priority,
            estimated_cost, sop_reference, status, created_at
        """
        with self._lock:
            self._counter += 1
            wo_id = f"WO-{self._counter:04d}"

        sop            = _SOP_MAP.get(failure_mode, _DEFAULT_SOP)
        estimated_cost = economic_data.get("total_loss", 0.0) if economic_data else 0.0

        order = {
            "work_order_id":  wo_id,
            "machine_id":     machine_id,
            "failure_mode":   failure_mode,
            "priority":       priority,
            "estimated_cost": round(estimated_cost, 2),
            "sop_reference":  sop,
            "status":         "open",
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._orders[wo_id] = order

        return order

    def get_order(self, work_order_id: str) -> dict | None:
        return self._orders.get(work_order_id)

    def list_orders(self) -> list[dict]:
        return list(self._orders.values())
