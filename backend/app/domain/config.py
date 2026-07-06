"""
Etapa 1.5 — Platform config store.

ConfigService provides typed get/set for platform_config table,
with a Redis cache (TTL=60s) that is invalidated on every PATCH.

Pydantic PlatformSettings model validates the full config on read
so callers get typed values, not raw JSONB.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import PlatformConfig
from app.infra.settings import settings

_logger = structlog.get_logger(__name__)
_CACHE_TTL = 60  # seconds
_CACHE_KEY = "config:platform"


# ── Pydantic schema for validation ───────────────────────────────────────────

class LLMSettings(BaseModel):
    model: str = "claude-sonnet-4-6"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=32768)
    price_input: float = Field(0.003, ge=0.0)
    price_output: float = Field(0.015, ge=0.0)


class RAGSettings(BaseModel):
    top_k: int = Field(5, ge=1, le=20)
    min_similarity: float = Field(0.65, ge=0.0, le=1.0)


class ThresholdSettings(BaseModel):
    failure: float = Field(0.85, ge=0.0, le=1.0)
    economic: float = Field(5000.0, ge=0.0)


class PlatformSettings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)


# ── Redis helper ──────────────────────────────────────────────────────────────

def _get_redis():
    import aioredis
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def _cache_get() -> dict | None:
    try:
        r = _get_redis()
        raw = await r.get(_CACHE_KEY)
        await r.aclose()
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def _cache_set(data: dict) -> None:
    try:
        r = _get_redis()
        await r.setex(_CACHE_KEY, _CACHE_TTL, json.dumps(data))
        await r.aclose()
    except Exception:
        pass


async def _cache_invalidate() -> None:
    try:
        r = _get_redis()
        await r.delete(_CACHE_KEY)
        await r.aclose()
    except Exception:
        pass


# ── Config service ────────────────────────────────────────────────────────────

async def get_all() -> dict[str, Any]:
    """
    Returns all platform_config rows as a flat dict {key: value}.
    Served from Redis cache when available; falls back to PostgreSQL.
    """
    cached = await _cache_get()
    if cached is not None:
        return cached

    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(select(PlatformConfig))).scalars().all()
            result = {r.key: (r.value.get("value") if r.value and "value" in r.value else r.value)
                      for r in rows}
            await _cache_set(result)
            return result
    except Exception:
        _logger.exception("config: get_all failed")
        return {}


async def get_typed() -> PlatformSettings:
    """Returns current config as a validated PlatformSettings instance."""
    flat = await get_all()
    return PlatformSettings(
        llm=LLMSettings(
            model=flat.get("llm.model", "claude-sonnet-4-6"),
            temperature=float(flat.get("llm.temperature", 0.1)),
            max_tokens=int(flat.get("llm.max_tokens", 4096)),
            price_input=float(flat.get("llm.price_input", 0.003)),
            price_output=float(flat.get("llm.price_output", 0.015)),
        ),
        rag=RAGSettings(
            top_k=int(flat.get("rag.top_k", 5)),
            min_similarity=float(flat.get("rag.min_similarity", 0.65)),
        ),
        thresholds=ThresholdSettings(
            failure=float(flat.get("thresholds.failure", 0.85)),
            economic=float(flat.get("thresholds.economic", 5000.0)),
        ),
    )


async def set_key(key: str, value: Any, updated_by: str = "api") -> None:
    """
    Update or insert a single config key and invalidate the Redis cache.
    value is stored as {"value": ...} JSONB envelope.
    """
    try:
        async with AsyncSessionLocal() as db:
            row = await db.get(PlatformConfig, key)
            if row:
                row.value = {"value": value}
                row.updated_by = updated_by
            else:
                db.add(PlatformConfig(key=key, value={"value": value}, updated_by=updated_by))
            await db.commit()
        await _cache_invalidate()
    except Exception:
        _logger.exception("config: set_key failed", key=key)
        raise


async def patch_many(updates: dict[str, Any], updated_by: str = "api") -> None:
    """Bulk update several keys at once, then invalidate cache once."""
    try:
        async with AsyncSessionLocal() as db:
            for key, value in updates.items():
                row = await db.get(PlatformConfig, key)
                if row:
                    row.value = {"value": value}
                    row.updated_by = updated_by
                else:
                    db.add(PlatformConfig(key=key, value={"value": value}, updated_by=updated_by))
            await db.commit()
        await _cache_invalidate()
    except Exception:
        _logger.exception("config: patch_many failed")
        raise
