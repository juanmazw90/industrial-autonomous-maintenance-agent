"""
Middleware de correlation_id.
Genera un UUID por request, lo propaga en contextvars de structlog
y lo devuelve como header X-Correlation-ID en la respuesta.
"""
from __future__ import annotations

import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
