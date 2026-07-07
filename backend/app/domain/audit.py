"""
Etapa 1.4 — Audit service + timeline helpers.

AuditService.record()   — writes an audit_log row (actor, action, entity, diff).
record_timeline_event() — writes a timeline_events row for a specific machine.

Both are fire-and-forget async functions designed to be awaited from routes
or scheduled via asyncio.create_task from sync contexts.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import AuditLog, TimelineEvent

_logger = structlog.get_logger(__name__)


async def record(
    action: str,
    entity_type: str,
    entity_id: str,
    actor_id: str | None = None,
    actor_type: str = "user",
    diff: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Persist an audit_log entry. Swallows exceptions so it never breaks a route."""
    try:
        async with AsyncSessionLocal() as db:
            db.add(AuditLog(
                id=str(uuid.uuid4()),
                actor_id=actor_id,
                actor_type=actor_type,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                diff=diff or {},
                correlation_id=correlation_id,
                created_at=datetime.now(UTC),
            ))
            await db.commit()
    except Exception:
        _logger.exception(
            "audit: record failed",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
        )


async def record_timeline_event(
    machine_id: str,
    kind: str,
    title: str,
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Persist a timeline_events row for a machine."""
    try:
        async with AsyncSessionLocal() as db:
            db.add(TimelineEvent(
                id=str(uuid.uuid4()),
                machine_id=machine_id,
                ts=datetime.now(UTC),
                kind=kind,
                title=title,
                payload=payload or {},
                correlation_id=correlation_id,
            ))
            await db.commit()
    except Exception:
        _logger.exception(
            "audit: record_timeline_event failed",
            machine_id=machine_id,
            kind=kind,
        )
