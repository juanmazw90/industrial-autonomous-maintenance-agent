"""Smoke test 0.1 — crear y consultar entidades via ORM."""
from __future__ import annotations

import uuid

import pytest
from app.infra.db.models import DemoUser, Line, Machine, Plant
from sqlalchemy import select


@pytest.mark.asyncio
async def test_create_demo_user(db):
    name = f"Ana Martínez {uuid.uuid4().hex[:6]}"
    user = DemoUser(name=name, role="supervisor")
    db.add(user)
    await db.flush()

    result = await db.execute(select(DemoUser).where(DemoUser.name == name))
    found = result.scalar_one()
    assert found.role == "supervisor"
    assert found.id is not None


@pytest.mark.asyncio
async def test_plant_line_machine_hierarchy(db):
    # Códigos únicos por corrida: la BD local compartida puede tener datos seed
    suffix = uuid.uuid4().hex[:6].upper()
    plant = Plant(code=f"PLANT-{suffix}", name="Plant Alpha")
    db.add(plant)
    await db.flush()

    line = Line(code=f"LINE-{suffix}", name="Production Line 1", plant_id=plant.id)
    db.add(line)
    await db.flush()

    machine = Machine(
        code=f"MCH-{suffix}", name="Compressor 01",
        machine_type="compressor", line_id=line.id, status="healthy"
    )
    db.add(machine)
    await db.flush()

    result = await db.execute(select(Machine).where(Machine.code == f"MCH-{suffix}"))
    m = result.scalar_one()
    assert m.status == "healthy"
    assert m.line_id == line.id
