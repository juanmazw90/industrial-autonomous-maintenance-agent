"""
economic_impact.py — Cálculo de impacto económico por fallo potencial.

Función pura sin estado ni I/O externo.
Modelo: pérdida = (OEE_nominal - OEE_efectivo) × tasa_producción × margen × horas
"""
from __future__ import annotations

_ECONOMICS: dict[str, dict] = {
    "centrifugal_pump": {
        "production_rate_per_hour": 120,
        "margin_per_unit": 4.50,
        "oee_baseline": 0.82,
    },
    "induction_motor": {
        "production_rate_per_hour": 200,
        "margin_per_unit": 2.80,
        "oee_baseline": 0.78,
    },
    "compressor": {
        "production_rate_per_hour": 85,
        "margin_per_unit": 8.20,
        "oee_baseline": 0.75,
    },
}

# Caída de OEE según nivel de alerta
_OEE_IMPACT: dict[str, float] = {
    "yellow": 0.15,
    "red":    0.35,
}


def calculate_loss(
    machine_id: str,
    machine_type: str,
    alert_level: str,
    hours_at_risk: float = 24.0,
) -> dict:
    """
    Calcula la pérdida económica estimada de una máquina en riesgo.

    Args:
        machine_id:    ID de la máquina (trazabilidad en el output)
        machine_type:  centrifugal_pump | induction_motor | compressor
        alert_level:   yellow | red
        hours_at_risk: horizonte de riesgo en horas (default 24)

    Returns:
        dict con hourly_loss, total_loss, oee_baseline, oee_effective,
        oee_impact, machine_id, machine_type, alert_level, hours_at_risk, currency
    """
    econ      = _ECONOMICS.get(machine_type, _ECONOMICS["centrifugal_pump"])
    oee_drop  = _OEE_IMPACT.get(alert_level, 0.15)
    oee_eff   = econ["oee_baseline"] * (1 - oee_drop)

    hourly_loss = (
        (econ["oee_baseline"] - oee_eff)
        * econ["production_rate_per_hour"]
        * econ["margin_per_unit"]
    )

    return {
        "machine_id":    machine_id,
        "machine_type":  machine_type,
        "alert_level":   alert_level,
        "hours_at_risk": hours_at_risk,
        "oee_baseline":  round(econ["oee_baseline"], 4),
        "oee_effective": round(oee_eff, 4),
        "oee_impact":    round(oee_drop, 4),
        "hourly_loss":   round(hourly_loss, 2),
        "total_loss":    round(hourly_loss * hours_at_risk, 2),
        "currency":      "USD",
    }
