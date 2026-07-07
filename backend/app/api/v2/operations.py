"""
Etapa 2.1 — Operations / Fleet / Assets routers.

GET /api/v2/operations/summary   — executive KPIs (single DB query)
GET /api/v2/fleet                — machine cards with live predictions
GET /api/v2/assets/{code}        — full asset overview
GET /api/v2/assets/{code}/timeline  — timeline events (filterable)
GET /api/v2/assets/{code}/sensors   — recent sensor readings (stub from predictor)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.db.base import get_db
from app.infra.db.models import (
    Alert,
    Line,
    Machine,
    ModelPrediction,
    Plant,
    TimelineEvent,
    WorkOrder,
)

router = APIRouter(prefix="/api/v2", tags=["operations"])


# ── /operations/summary ────────────────────────────────────────────────────────

@router.get("/operations/summary")
async def operations_summary(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Executive KPIs: plant health score, fleet state, alerts, work orders, AI stats.
    Single-shot aggregation — drives the Executive Operations Center dashboard.
    """
    # Machine counts by status
    machine_counts = (await db.execute(
        select(Machine.status, func.count(Machine.id).label("n"))
        .group_by(Machine.status)
    )).all()
    status_map = {row.status: row.n for row in machine_counts}
    total_machines = sum(status_map.values())
    healthy = status_map.get("healthy", 0)
    warning = status_map.get("warning", 0)
    critical = status_map.get("critical", 0)

    # Plant health score: healthy=100%, warning=50%, critical=0%
    if total_machines:
        health_score = round((healthy * 100 + warning * 50) / total_machines, 1)
    else:
        health_score = 100.0

    # Alerts by status
    alert_counts = (await db.execute(
        select(Alert.status, func.count(Alert.id).label("n"))
        .group_by(Alert.status)
    )).all()
    alert_map = {row.status: row.n for row in alert_counts}

    alerts_new = alert_map.get("new", 0)
    alerts_open = sum(alert_map.get(s, 0) for s in ("new", "acknowledged", "assigned"))

    # Severity counts for open alerts
    sev_counts = (await db.execute(
        select(Alert.severity, func.count(Alert.id).label("n"))
        .where(Alert.status.in_(["new", "acknowledged", "assigned"]))
        .group_by(Alert.severity)
    )).all()
    sev_map = {row.severity: row.n for row in sev_counts}

    # Work orders
    wo_counts = (await db.execute(
        select(WorkOrder.status, func.count(WorkOrder.id).label("n"))
        .group_by(WorkOrder.status)
    )).all()
    wo_map = {row.status: row.n for row in wo_counts}
    wo_open = wo_map.get("open", 0)
    wo_total = sum(wo_map.values())

    # Economic risk: sum estimated_cost of open WOs
    risk_row = (await db.execute(
        select(func.coalesce(func.sum(WorkOrder.estimated_cost), 0.0))
        .where(WorkOrder.status.in_(["open", "assigned", "in_progress"]))
    )).scalar()
    risk_usd = round(float(risk_row or 0.0), 2)

    # Latest predictions — avg failure probability
    latest_preds = (await db.execute(
        select(ModelPrediction.probability, ModelPrediction.machine_id)
        .where(ModelPrediction.model_name == "amia-failure-model")
        .distinct(ModelPrediction.machine_id)
        .order_by(ModelPrediction.machine_id, ModelPrediction.created_at.desc())
    )).all()
    avg_failure_prob = (
        round(sum(r.probability or 0.0 for r in latest_preds) / len(latest_preds) * 100, 1)
        if latest_preds else 0.0
    )

    return {
        "plant_health_score": health_score,
        "machines": {
            "total": total_machines,
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
        },
        "alerts": {
            "open": alerts_open,
            "new": alerts_new,
            "critical": sev_map.get("critical", 0),
            "high": sev_map.get("high", 0),
            "medium": sev_map.get("medium", 0),
            "low": sev_map.get("low", 0),
        },
        "work_orders": {
            "open": wo_open,
            "total": wo_total,
        },
        "risk_exposure_usd": risk_usd,
        "avg_failure_probability_pct": avg_failure_prob,
    }


# ── /fleet ─────────────────────────────────────────────────────────────────────

@router.get("/fleet")
async def get_fleet(
    status: str | None = Query(None, description="Filter by status: healthy|warning|critical"),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Machine cards with latest prediction, open alert count, and RUL stub."""
    q = select(Machine, Line, Plant).join(Line, Machine.line_id == Line.id).join(Plant, Line.plant_id == Plant.id)
    if status:
        q = q.where(Machine.status == status)
    rows = (await db.execute(q)).all()

    result = []
    for machine, line, plant in rows:
        # Latest failure prediction
        pred_row = (await db.execute(
            select(ModelPrediction)
            .where(
                ModelPrediction.machine_id == machine.id,
                ModelPrediction.model_name == "amia-failure-model",
            )
            .order_by(ModelPrediction.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()

        # Open alert count
        alert_n = (await db.execute(
            select(func.count(Alert.id))
            .where(Alert.machine_id == machine.id, Alert.status != "resolved")
        )).scalar() or 0

        result.append({
            "id": machine.id,
            "code": machine.code,
            "name": machine.name,
            "type": machine.machine_type,
            "status": machine.status,
            "line": {"code": line.code, "name": line.name},
            "plant": {"code": plant.code, "name": plant.name},
            "failure_probability": pred_row.probability if pred_row else None,
            "prediction_at": pred_row.created_at.isoformat() if pred_row else None,
            "open_alerts": alert_n,
        })

    return result


# ── /assets/{code} ─────────────────────────────────────────────────────────────

async def _get_machine_by_code(db, code: str) -> Machine:
    machine = (await db.execute(
        select(Machine).where(Machine.code == code.upper())
    )).scalar_one_or_none()
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine '{code}' not found")
    return machine


@router.get("/assets/{code}")
async def get_asset(code: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Full asset overview: machine metadata + latest prediction + open alerts + open WOs."""
    machine = await _get_machine_by_code(db, code)

    # Latest prediction
    pred = (await db.execute(
        select(ModelPrediction)
        .where(ModelPrediction.machine_id == machine.id)
        .order_by(ModelPrediction.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    # Open alerts
    alerts = (await db.execute(
        select(Alert)
        .where(Alert.machine_id == machine.id, Alert.status != "resolved")
        .order_by(Alert.created_at.desc())
        .limit(10)
    )).scalars().all()

    # Open work orders
    work_orders = (await db.execute(
        select(WorkOrder)
        .where(WorkOrder.machine_id == machine.id, WorkOrder.status != "completed")
        .order_by(WorkOrder.created_at.desc())
        .limit(5)
    )).scalars().all()

    return {
        "id": machine.id,
        "code": machine.code,
        "name": machine.name,
        "type": machine.machine_type,
        "status": machine.status,
        "latest_prediction": {
            "id": pred.id,
            "probability": pred.probability,
            "prediction": pred.prediction,
            "shap_top_features": pred.shap_top_features,
            "model_name": pred.model_name,
            "model_version": pred.model_version,
            "created_at": pred.created_at.isoformat(),
        } if pred else None,
        "open_alerts": [
            {
                "id": a.id,
                "severity": a.severity,
                "status": a.status,
                "title": a.title,
                "created_at": a.created_at.isoformat(),
            }
            for a in alerts
        ],
        "open_work_orders": [
            {
                "id": wo.id,
                "title": wo.title,
                "priority": wo.priority,
                "status": wo.status,
                "estimated_cost": wo.estimated_cost,
            }
            for wo in work_orders
        ],
    }


@router.get("/timeline")
async def get_global_timeline(
    kind: str | None = Query(None, description="Event kind filter"),
    machine_code: str | None = Query(None, description="Filter by machine code"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Cross-plant timeline: all events from all machines, newest first."""
    q = (
        select(TimelineEvent, Machine)
        .join(Machine, TimelineEvent.machine_id == Machine.id)
        .order_by(TimelineEvent.ts.desc())
    )
    count_q = select(func.count(TimelineEvent.id))

    if kind:
        q = q.where(TimelineEvent.kind == kind)
        count_q = count_q.where(TimelineEvent.kind == kind)
    if machine_code:
        q = q.where(Machine.code == machine_code.upper())
        count_q = count_q.join(Machine, TimelineEvent.machine_id == Machine.id).where(
            Machine.code == machine_code.upper()
        )

    total = (await db.execute(count_q)).scalar() or 0
    rows = (await db.execute(q.limit(limit).offset(offset))).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": [
            {
                "id": e.id,
                "ts": e.ts.isoformat(),
                "kind": e.kind,
                "title": e.title,
                "payload": e.payload,
                "correlation_id": e.correlation_id,
                "machine_code": m.code,
                "machine_name": m.name,
            }
            for e, m in rows
        ],
    }


@router.get("/assets/{code}/timeline")
async def get_asset_timeline(
    code: str,
    kind: str | None = Query(None, description="Event kind filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Timeline events for a machine, newest first."""
    machine = await _get_machine_by_code(db, code)

    q = (
        select(TimelineEvent)
        .where(TimelineEvent.machine_id == machine.id)
        .order_by(TimelineEvent.ts.desc())
    )
    if kind:
        q = q.where(TimelineEvent.kind == kind)

    total = (await db.execute(
        select(func.count(TimelineEvent.id))
        .where(TimelineEvent.machine_id == machine.id)
    )).scalar() or 0

    events = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    return {
        "machine_code": code.upper(),
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": [
            {
                "id": e.id,
                "ts": e.ts.isoformat(),
                "kind": e.kind,
                "title": e.title,
                "payload": e.payload,
                "correlation_id": e.correlation_id,
            }
            for e in events
        ],
    }


@router.get("/assets/{code}/sensors")
async def get_asset_sensors(code: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Returns the latest feature vector for a machine (from predictor in-memory state).
    Used by Asset Detail > Sensors tab for current reading values.
    """

    machine = await _get_machine_by_code(db, code)

    # Access the singleton predictor via the app state (imported at module level)
    # For now: return the prediction from the latest model_prediction row.
    pred = (await db.execute(
        select(ModelPrediction)
        .where(
            ModelPrediction.machine_id == machine.id,
            ModelPrediction.model_name == "amia-failure-model",
        )
        .order_by(ModelPrediction.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    return {
        "machine_code": code.upper(),
        "latest_reading_at": pred.created_at.isoformat() if pred else None,
        "prediction": pred.prediction if pred else {},
        "shap_features": pred.shap_top_features if pred else [],
    }
