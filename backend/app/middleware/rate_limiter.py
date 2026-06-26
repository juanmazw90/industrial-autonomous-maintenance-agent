"""
Rate limiting por IP usando Redis con ventana deslizante de 60 segundos.
Endpoints limitados: /process_input y /predict/*.
"""

import time

import redis.asyncio as aioredis
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_RATE_LIMITED_PREFIXES = ("/process_input", "/predict/")
_DEFAULT_LIMIT  = 10   # requests
_DEFAULT_WINDOW = 60   # seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = "redis://localhost:6379", limit: int = _DEFAULT_LIMIT, window: int = _DEFAULT_WINDOW):
        super().__init__(app)
        self._redis   = aioredis.from_url(redis_url, decode_responses=True)
        self._limit   = limit
        self._window  = window

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not any(path.startswith(p) for p in _RATE_LIMITED_PREFIXES):
            return await call_next(request)

        ip  = request.client.host if request.client else "unknown"
        key = f"rate:{ip}:{int(time.time()) // self._window}"

        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, self._window)
        except Exception:
            # Si Redis no está disponible, dejamos pasar la request
            return await call_next(request)

        if count > self._limit:
            retry_after = self._window - (int(time.time()) % self._window)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Demasiadas solicitudes. Límite: {self._limit} por {self._window}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
