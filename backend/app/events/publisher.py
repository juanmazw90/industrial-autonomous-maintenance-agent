"""
Event bus: publica eventos a Redis pub/sub canal 'events:live'.
El endpoint SSE /api/v2/events/stream consume este canal y reenvía al frontend.
"""
from __future__ import annotations

import json

import redis.asyncio as aioredis

from app.events.schemas import BaseEvent
from app.infra.settings import settings

_CHANNEL = "events:live"

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def publish(event: BaseEvent) -> None:
    """Publica un evento serializado como JSON en el canal events:live."""
    r = _get_redis()
    await r.publish(_CHANNEL, event.model_dump_json())


async def event_stream():
    """Generador async para el endpoint SSE. Consume eventos del canal Redis."""
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe(_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                yield f"data: {message['data']}\n\n"
    finally:
        await pubsub.unsubscribe(_CHANNEL)
        await r.aclose()
