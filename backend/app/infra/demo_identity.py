"""
Identidad demo (ADR-01): sustituye auth real por un selector de rol simulado.
El middleware resuelve el header X-Demo-User contra demo_users en PostgreSQL.
Sin header → actor system.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

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


class DemoIdentityMiddleware:
    """Inyecta scope['state']['actor'] a partir del header X-Demo-User."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            raw_headers = {k.lower(): v for k, v in scope.get("headers", [])}
            user_id_bytes = raw_headers.get(b"x-demo-user", b"")
            user_id = user_id_bytes.decode() if user_id_bytes else None

            if user_id:
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(DemoUser).where(DemoUser.id == user_id)
                    )
                    user = result.scalar_one_or_none()
                actor = Actor(id=user.id, name=user.name, role=user.role) if user else SYSTEM_ACTOR
            else:
                actor = SYSTEM_ACTOR

            scope.setdefault("state", {})["actor"] = actor

        await self.app(scope, receive, send)
