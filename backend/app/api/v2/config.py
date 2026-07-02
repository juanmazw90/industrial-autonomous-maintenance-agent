"""
Etapa 1.5 — Config API router.

GET  /api/v2/config          → full config as typed PlatformSettings
PATCH /api/v2/config         → bulk update; validated, cached, audited
GET  /api/v2/config/{key}   → single key value
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, model_validator

from app.domain import audit, config as cfg

router = APIRouter(prefix="/api/v2/config", tags=["config"])


class ConfigPatch(BaseModel):
    """Flat key→value patch dict. Keys must match known platform_config keys."""
    updates: dict[str, Any]

    @model_validator(mode="after")
    def validate_keys(self) -> "ConfigPatch":
        allowed_prefixes = ("llm.", "rag.", "thresholds.")
        invalid = [k for k in self.updates if not any(k.startswith(p) for p in allowed_prefixes)]
        if invalid:
            raise ValueError(f"Unknown config keys: {invalid}. Allowed prefixes: {allowed_prefixes}")
        return self


@router.get("")
async def get_config(request: Request):
    """Returns the full platform configuration (Redis-cached, TTL=60s)."""
    return (await cfg.get_typed()).model_dump()


@router.get("/{key}")
async def get_config_key(key: str):
    """Returns a single config value by key."""
    all_cfg = await cfg.get_all()
    if key not in all_cfg:
        raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
    return {"key": key, "value": all_cfg[key]}


@router.patch("")
async def patch_config(body: ConfigPatch, request: Request):
    """
    Update one or more config keys.
    Validated by PlatformSettings on read; invalidates Redis cache immediately.
    """
    actor = getattr(getattr(request, "state", None), "actor", None)
    actor_id = getattr(actor, "id", None)
    updated_by = getattr(actor, "name", "api") if actor else "api"

    await cfg.patch_many(body.updates, updated_by=updated_by)

    # Audit the change.
    await audit.record(
        action="patch",
        entity_type="platform_config",
        entity_id="platform_config",
        actor_id=actor_id,
        actor_type="user" if actor_id else "system",
        diff={"updates": body.updates},
        correlation_id=getattr(getattr(request, "state", None), "correlation_id", None),
    )

    return {"status": "ok", "updated_keys": list(body.updates.keys())}
