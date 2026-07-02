"""
Etapa 1.4 — Audit middleware.

Intercepts every mutating request (POST / PATCH / DELETE / PUT) and
writes an audit_log entry with actor (from X-Demo-User / request.state.actor),
HTTP action, path (used as entity_type + entity_id), correlation_id and
a summary diff extracted from the request body.

This middleware runs AFTER CorrelationIdMiddleware so correlation_id is set.
"""
from __future__ import annotations

import json

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.domain.audit import record

_logger = structlog.get_logger(__name__)

_AUDIT_METHODS = {"POST", "PATCH", "PUT", "DELETE"}

# Paths that should not be audited (read-only or infrastructure).
_SKIP_PREFIXES = (
    "/health",
    "/metrics",
    "/api/v2/events",
    "/docs",
    "/openapi",
    "/redoc",
)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Captures mutating requests and records them to audit_log.
    Body is read once and re-injected so downstream handlers still see it.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method not in _AUDIT_METHODS:
            return await call_next(request)
        if any(request.url.path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        # Read body once — stash for re-consumption.
        body_bytes = await request.body()
        body_summary: dict = {}
        if body_bytes:
            try:
                body_summary = json.loads(body_bytes)
                if isinstance(body_summary, dict) and len(str(body_summary)) > 500:
                    body_summary = {k: v for k, v in list(body_summary.items())[:10]}
            except Exception:
                body_summary = {"raw_length": len(body_bytes)}

        # Re-inject body so the actual route handler can read it.
        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]

        response = await call_next(request)

        # Only audit successful mutations (2xx).
        if 200 <= response.status_code < 300:
            actor = getattr(getattr(request, "state", None), "actor", None)
            actor_id = getattr(actor, "id", None)
            actor_type = "user" if actor_id else "system"

            correlation_id = getattr(getattr(request, "state", None), "correlation_id", None)

            path_parts = request.url.path.strip("/").split("/")
            entity_type = path_parts[-2] if len(path_parts) >= 2 else path_parts[0]
            entity_id = path_parts[-1] if len(path_parts) >= 2 else "unknown"

            import asyncio
            asyncio.create_task(record(
                action=request.method.lower(),
                entity_type=entity_type,
                entity_id=entity_id,
                actor_id=actor_id,
                actor_type=actor_type,
                diff={"method": request.method, "path": request.url.path, "body": body_summary},
                correlation_id=correlation_id,
            ))

        return response
