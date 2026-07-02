"""
Identidad demo (ADR-01): sustituye auth real por un selector de rol simulado.
El middleware resuelve el header X-Demo-User contra demo_users en PostgreSQL.
Sin header → actor system.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import DemoUser


@dataclass(frozen=True)
class Actor:
    id: str | None
    name: str
    role: str


SYSTEM_ACTOR = Actor(id=None, name="system", role="system")

# Roles disponibles en el seeder de demo
DEMO_ROLES = (
    "operator",
    "supervisor",
    "maintenance_manager",
    "plant_director",
    "ai_engineer",
)


class DemoIdentityMiddleware(BaseHTTPMiddleware):
    """Inyecta request.state.actor a partir del header X-Demo-User."""

    async def dispatch(self, request: Request, call_next) -> Response:
        user_id = request.headers.get("X-Demo-User")
        if user_id:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(DemoUser).where(DemoUser.id == user_id)
                )
                user = result.scalar_one_or_none()
            if user:
                request.state.actor = Actor(id=user.id, name=user.name, role=user.role)
            else:
                request.state.actor = SYSTEM_ACTOR
        else:
            request.state.actor = SYSTEM_ACTOR
        return await call_next(request)
