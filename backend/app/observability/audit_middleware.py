"""
Etapa 1.4 — Audit middleware.

Intercepts every mutating request (POST / PATCH / DELETE / PUT) and
writes an audit_log entry with actor (from X-Demo-User / scope state),
HTTP action, path (used as entity_type + entity_id), correlation_id and
a summary diff extracted from the request body.
"""
from __future__ import annotations

import asyncio
import json

import structlog

from app.domain.audit import record

_logger = structlog.get_logger(__name__)

_AUDIT_METHODS = {"POST", "PATCH", "PUT", "DELETE"}

_SKIP_PREFIXES = (
    "/health",
    "/metrics",
    "/api/v2/events",
    "/docs",
    "/openapi",
    "/redoc",
)


class AuditMiddleware:
    """
    Captures mutating requests and records them to audit_log.
    Pure ASGI implementation — compatible with StreamingResponse.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")

        if method not in _AUDIT_METHODS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Drain the receive channel to buffer the full request body.
        body_bytes = b""
        more_body = True
        while more_body:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                await send({"type": "http.response.start", "status": 400, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            body_bytes += msg.get("body", b"")
            more_body = msg.get("more_body", False)

        body_summary: dict = {}
        if body_bytes:
            try:
                body_summary = json.loads(body_bytes)
                if isinstance(body_summary, dict) and len(str(body_summary)) > 500:
                    body_summary = {k: v for k, v in list(body_summary.items())[:10]}
            except Exception:
                body_summary = {"raw_length": len(body_bytes)}

        # Replay receive: return buffered body once, then forward to the real
        # receive so StreamingResponse's disconnect listener can block correctly.
        _consumed = False

        async def replay_receive():
            nonlocal _consumed
            if not _consumed:
                _consumed = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return await receive()

        # Intercept send to record audit on 2xx responses.
        async def audit_send(message):
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                if 200 <= status_code < 300:
                    state = scope.get("state", {})
                    actor = state.get("actor", None)
                    actor_id = getattr(actor, "id", None)
                    actor_type = "user" if actor_id else "system"
                    correlation_id = state.get("correlation_id", None)

                    path_parts = path.strip("/").split("/")
                    entity_type = path_parts[-2] if len(path_parts) >= 2 else path_parts[0]
                    entity_id = path_parts[-1] if len(path_parts) >= 2 else "unknown"

                    asyncio.create_task(record(
                        action=method.lower(),
                        entity_type=entity_type,
                        entity_id=entity_id,
                        actor_id=actor_id,
                        actor_type=actor_type,
                        diff={"method": method, "path": path, "body": body_summary},
                        correlation_id=correlation_id,
                    ))
            await send(message)

        await self.app(scope, replay_receive, audit_send)
