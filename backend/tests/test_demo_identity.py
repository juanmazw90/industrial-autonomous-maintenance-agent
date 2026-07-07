"""Tests Etapa 0.2 — identidad demo."""
from __future__ import annotations

import pytest
from app.infra.db.models import DemoUser
from app.infra.demo_identity import SYSTEM_ACTOR, Actor
from sqlalchemy import select


@pytest.mark.asyncio
async def test_system_actor_is_default():
    assert SYSTEM_ACTOR.id is None
    assert SYSTEM_ACTOR.role == "system"


@pytest.mark.asyncio
async def test_create_and_resolve_demo_user(db):
    import uuid
    unique_name = f"Sofía López {uuid.uuid4().hex[:6]}"
    user = DemoUser(name=unique_name, role="plant_director")
    db.add(user)
    await db.commit()

    result = await db.execute(select(DemoUser).where(DemoUser.id == user.id))
    found = result.scalar_one()
    assert found.role == "plant_director"

    actor = Actor(id=found.id, name=found.name, role=found.role)
    assert actor.id == found.id
    assert actor.role == "plant_director"
