"""Smoke test 0.1 — crear y consultar entidades via ORM."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.infra.db.models import DemoUser, Machine, Plant, Line


@pytest.mark.asyncio
async def test_create_demo_user(db):
    user = DemoUser(name="Ana Martínez", role="supervisor")
    db.add(user)
    await db.flush()

    result = await db.execute(select(DemoUser).where(DemoUser.name == "Ana Martínez"))
    found = result.scalar_one()
    assert found.role == "supervisor"
    assert found.id is not None


@pytest.mark.asyncio
async def test_plant_line_machine_hierarchy(db):
    plant = Plant(code="PLANT-A", name="Plant Alpha")
    db.add(plant)
    await db.flush()

    line = Line(code="LINE-01", name="Production Line 1", plant_id=plant.id)
    db.add(line)
    await db.flush()

    machine = Machine(
        code="MCH-01", name="Compressor 01",
        machine_type="compressor", line_id=line.id, status="healthy"
    )
    db.add(machine)
    await db.flush()

    result = await db.execute(select(Machine).where(Machine.code == "MCH-01"))
    m = result.scalar_one()
    assert m.status == "healthy"
    assert m.line_id == line.id
