"""
Etapa 2.4 — Platform Operations routers.

GET/POST  /api/v2/alerts               — list + manual create
PATCH     /api/v2/alerts/{id}          — status transitions
GET/POST  /api/v2/work-orders          — list + create
PATCH     /api/v2/work-orders/{id}     — update status / assignment
GET       /api/v2/logs                 — structured log explorer (last N lines from structlog)
GET       /api/v2/audit                — audit trail
GET       /api/v2/monitoring/services  — active service health checks
"""
from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import alerting
from app.infra.db.base import AsyncSessionLocal, get_db
from app.infra.db.models import Alert, AuditLog, Machine, WorkOrder
from app.infra.settings import settings

_logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["platform"])


# ── /alerts ────────────────────────────────────────────────────────────────────

class AlertCreate(BaseModel):
    machine_code: str
    severity: str  # critical|high|medium|low
    title: str
    description: str | None = None
    source: str = "manual"


class AlertTransition(BaseModel):
    status: str
    actor_id: str | None = None
    assigned_to: str | None = None


@router.get("/alerts")
async def list_alerts(
    status: str | None = Query(None),
    severity: str | None = Query(None),
    machine_code: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    q = select(Alert, Machine).join(Machine, Alert.machine_id == Machine.id)
    count_q = select(func.count(Alert.id))

    if status:
        q = q.where(Alert.status == status)
        count_q = count_q.where(Alert.status == status)
    if severity:
        q = q.where(Alert.severity == severity)
        count_q = count_q.where(Alert.severity == severity)
    if machine_code:
        machine = (await db.execute(
            select(Machine).where(Machine.code == machine_code.upper())
        )).scalar_one_or_none()
        if machine:
            q = q.where(Alert.machine_id == machine.id)
            count_q = count_q.where(Alert.machine_id == machine.id)

    q = q.order_by(Alert.created_at.desc())
    total = (await db.execute(count_q)).scalar() or 0
    rows = (await db.execute(q.limit(limit).offset(offset))).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": [
            {
                "id": a.id,
                "machine_code": m.code,
                "machine_name": m.name,
                "severity": a.severity,
                "status": a.status,
                "source": a.source,
                "title": a.title,
                "description": a.description,
                "created_at": a.created_at.isoformat(),
                "acknowledged_by": a.acknowledged_by,
                "assigned_to": a.assigned_to,
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            }
            for a, m in rows
        ],
    }


@router.post("/alerts", status_code=201)
async def create_alert(body: AlertCreate, request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    machine = (await db.execute(
        select(Machine).where(Machine.code == body.machine_code.upper())
    )).scalar_one_or_none()
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine '{body.machine_code}' not found")

    if body.severity not in ("critical", "high", "medium", "low"):
        raise HTTPException(status_code=422, detail=f"Invalid severity '{body.severity}'")

    alert = Alert(
        machine_id=machine.id,
        severity=body.severity,
        source=body.source,
        title=body.title,
        description=body.description,
        status="new",
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    return {
        "id": alert.id,
        "machine_code": body.machine_code.upper(),
        "severity": alert.severity,
        "status": alert.status,
        "title": alert.title,
        "created_at": alert.created_at.isoformat(),
    }


@router.patch("/alerts/{alert_id}")
async def transition_alert(
    alert_id: str, body: AlertTransition, request: Request
) -> dict[str, Any]:
    actor = getattr(getattr(request, "state", None), "actor", None)
    actor_id = body.actor_id or getattr(actor, "id", None)

    updated = await alerting.transition(
        alert_id=alert_id,
        new_status=body.status,
        actor_id=actor_id,
        assigned_to=body.assigned_to,
    )
    if not updated:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot transition alert '{alert_id}' to '{body.status}'",
        )
    return {
        "id": updated.id,
        "status": updated.status,
        "acknowledged_by": updated.acknowledged_by,
        "assigned_to": updated.assigned_to,
        "resolved_at": updated.resolved_at.isoformat() if updated.resolved_at else None,
    }


# ── /work-orders ───────────────────────────────────────────────────────────────

class WorkOrderCreate(BaseModel):
    machine_code: str
    title: str
    description: str | None = None
    priority: str = "medium"  # critical|high|medium|low
    estimated_cost: float | None = None


class WorkOrderUpdate(BaseModel):
    status: str | None = None
    assigned_to: str | None = None
    estimated_cost: float | None = None


@router.get("/work-orders")
async def list_work_orders_v2(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    machine_code: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    q = select(WorkOrder, Machine).join(Machine, WorkOrder.machine_id == Machine.id)
    count_q = select(func.count(WorkOrder.id))

    if status:
        q = q.where(WorkOrder.status == status)
        count_q = count_q.where(WorkOrder.status == status)
    if priority:
        q = q.where(WorkOrder.priority == priority)
        count_q = count_q.where(WorkOrder.priority == priority)
    if machine_code:
        machine = (await db.execute(
            select(Machine).where(Machine.code == machine_code.upper())
        )).scalar_one_or_none()
        if machine:
            q = q.where(WorkOrder.machine_id == machine.id)
            count_q = count_q.where(WorkOrder.machine_id == machine.id)

    q = q.order_by(WorkOrder.created_at.desc())
    total = (await db.execute(count_q)).scalar() or 0
    rows = (await db.execute(q.limit(limit).offset(offset))).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "work_orders": [
            {
                "id": wo.id,
                "machine_code": m.code,
                "machine_name": m.name,
                "title": wo.title,
                "description": wo.description,
                "priority": wo.priority,
                "status": wo.status,
                "assigned_to": wo.assigned_to,
                "estimated_cost": wo.estimated_cost,
                "alert_id": wo.alert_id,
                "created_at": wo.created_at.isoformat(),
                "updated_at": wo.updated_at.isoformat() if wo.updated_at else None,
            }
            for wo, m in rows
        ],
    }


@router.post("/work-orders", status_code=201)
async def create_work_order(
    body: WorkOrderCreate, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    if body.priority not in ("critical", "high", "medium", "low"):
        raise HTTPException(status_code=422, detail=f"Invalid priority '{body.priority}'")

    machine = (await db.execute(
        select(Machine).where(Machine.code == body.machine_code.upper())
    )).scalar_one_or_none()
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine '{body.machine_code}' not found")

    wo = WorkOrder(
        machine_id=machine.id,
        title=body.title,
        description=body.description,
        priority=body.priority,
        status="open",
        estimated_cost=body.estimated_cost,
    )
    db.add(wo)
    await db.commit()
    await db.refresh(wo)

    return {
        "id": wo.id,
        "machine_code": body.machine_code.upper(),
        "title": wo.title,
        "priority": wo.priority,
        "status": wo.status,
        "estimated_cost": wo.estimated_cost,
        "created_at": wo.created_at.isoformat(),
    }


@router.patch("/work-orders/{wo_id}")
async def update_work_order(wo_id: str, body: WorkOrderUpdate, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    _VALID_WO_STATUSES = ("open", "assigned", "in_progress", "completed")
    if body.status and body.status not in _VALID_WO_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status '{body.status}'")

    wo = (await db.execute(
        select(WorkOrder).where(WorkOrder.id == wo_id)
    )).scalar_one_or_none()
    if not wo:
        raise HTTPException(status_code=404, detail=f"WorkOrder '{wo_id}' not found")

    if body.status:
        wo.status = body.status
    if body.assigned_to:
        wo.assigned_to = body.assigned_to
    if body.estimated_cost is not None:
        wo.estimated_cost = body.estimated_cost

    await db.commit()
    await db.refresh(wo)

    return {
        "id": wo.id,
        "status": wo.status,
        "assigned_to": wo.assigned_to,
        "estimated_cost": wo.estimated_cost,
        "updated_at": wo.updated_at.isoformat() if wo.updated_at else None,
    }


# ── /logs ──────────────────────────────────────────────────────────────────────

@router.get("/logs")
async def get_logs(
    limit: int = Query(100, ge=1, le=1000),
    level: str | None = Query(None, description="Filter: debug|info|warning|error|critical"),
) -> dict[str, Any]:
    """
    Returns recent structured log lines from the application log file.
    Falls back to an empty list when the log file is not configured.
    """
    log_file = settings.log_file_path
    entries: list[dict] = []

    if log_file and os.path.exists(log_file):
        import json

        lines: list[str] = []
        try:
            with open(log_file) as f:
                lines = f.readlines()
        except OSError:
            pass

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                entry = {"message": line}

            if level and entry.get("level", "").lower() != level.lower():
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break

    return {
        "source": log_file or "not_configured",
        "count": len(entries),
        "entries": entries,
    }


# ── /audit ─────────────────────────────────────────────────────────────────────

@router.get("/audit")
async def get_audit_trail(
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
    actor_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    q = select(AuditLog).order_by(AuditLog.created_at.desc())
    count_q = select(func.count(AuditLog.id))

    if entity_type:
        q = q.where(AuditLog.entity_type == entity_type)
        count_q = count_q.where(AuditLog.entity_type == entity_type)
    if entity_id:
        q = q.where(AuditLog.entity_id == entity_id)
        count_q = count_q.where(AuditLog.entity_id == entity_id)
    if actor_id:
        q = q.where(AuditLog.actor_id == actor_id)
        count_q = count_q.where(AuditLog.actor_id == actor_id)

    total = (await db.execute(count_q)).scalar() or 0
    rows = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": [
            {
                "id": r.id,
                "action": r.action,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "actor_id": r.actor_id,
                "actor_type": r.actor_type,
                "diff": r.diff,
                "correlation_id": r.correlation_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }


# ── /monitoring/services ───────────────────────────────────────────────────────

async def _check(name: str, coro) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(coro, timeout=3.0)
        return {"name": name, "status": "ok", "detail": result}
    except TimeoutError:
        return {"name": name, "status": "timeout"}
    except Exception as exc:
        return {"name": name, "status": "error", "detail": str(exc)[:200]}


async def _check_postgres() -> str:
    async with AsyncSessionLocal() as db:
        await db.execute(select(func.now()))
    return "connected"


async def _check_redis() -> str:
    import redis.asyncio as aioredis

    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    pong = await r.ping()
    await r.aclose()
    return "pong" if pong else "no response"


async def _check_qdrant() -> str:
    from qdrant_client import AsyncQdrantClient

    from app.services.rag_config import RAGConfig
    cfg = RAGConfig()
    qdrant_url = f"http://{cfg.qdrant_host}:{cfg.qdrant_port}"
    client = AsyncQdrantClient(url=qdrant_url)
    info = await client.get_collections()
    await client.close()
    return f"{len(info.collections)} collections"


@router.get("/monitoring/services")
async def monitoring_services() -> dict[str, Any]:
    """Active health checks for all backend services."""
    checks = await asyncio.gather(
        _check("postgres", _check_postgres()),
        _check("redis", _check_redis()),
        _check("qdrant", _check_qdrant()),
        return_exceptions=False,
    )
    all_ok = all(c["status"] == "ok" for c in checks)
    return {
        "overall": "ok" if all_ok else "degraded",
        "services": list(checks),
        "checked_at": datetime.now(UTC).isoformat(),
    }
