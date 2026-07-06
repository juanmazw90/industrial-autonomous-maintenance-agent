"""
Middleware de correlation_id.
Genera un UUID por request, lo propaga en contextvars de structlog
y lo devuelve como header X-Correlation-ID en la respuesta.
"""
from __future__ import annotations

import uuid

import structlog


class CorrelationIdMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = {k.lower(): v for k, v in scope.get("headers", [])}
        correlation_id = (
            raw_headers.get(b"x-correlation-id", b"").decode() or str(uuid.uuid4())
        )

        scope.setdefault("state", {})["correlation_id"] = correlation_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=scope.get("path", ""),
            method=scope.get("method", ""),
        )

        async def _send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, _send)
