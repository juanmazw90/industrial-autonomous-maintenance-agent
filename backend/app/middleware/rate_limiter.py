"""
Rate limiting por IP usando Redis con ventana deslizante de 60 segundos.
Endpoints limitados: /process_input y /predict/*.
Pure ASGI implementation — compatible con StreamingResponse.
"""

import time

import redis.asyncio as aioredis
from fastapi.responses import JSONResponse

_RATE_LIMITED_PREFIXES = ("/process_input", "/predict/")
_DEFAULT_LIMIT  = 10   # requests
_DEFAULT_WINDOW = 60   # seconds


class RateLimitMiddleware:
    def __init__(self, app, redis_url: str = "redis://localhost:6379", limit: int = _DEFAULT_LIMIT, window: int = _DEFAULT_WINDOW):
        self.app    = app
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._limit  = limit
        self._window = window

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not any(path.startswith(p) for p in _RATE_LIMITED_PREFIXES):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        ip  = client[0] if client else "unknown"
        key = f"rate:{ip}:{int(time.time()) // self._window}"

        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, self._window)
        except Exception:
            await self.app(scope, receive, send)
            return

        if count > self._limit:
            retry_after = self._window - (int(time.time()) % self._window)
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": f"Demasiadas solicitudes. Límite: {self._limit} por {self._window}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
