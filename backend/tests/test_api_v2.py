"""
Tests de endpoints /api/v2 usando dependency override de get_db.

Posibles gracias a la inyección de dependencias: los routers reciben la sesión
vía Depends(get_db) y aquí se sustituye por una sesión de test. Usa datos con
códigos únicos por corrida (la BD local compartida puede tener seed demo).
"""
from __future__ import annotations

import uuid

import pytest_asyncio
from app.infra.db.base import get_db
from app.infra.db.models import Line, Machine, Plant
from app.main import app
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.conftest import TEST_DATABASE_URL


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # la factory queda disponible para seeds dentro de los tests
        c.session_factory = factory  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


async def _seed_machine(factory) -> str:
    """Crea plant→line→machine con códigos únicos y devuelve el código de máquina."""
    suffix = uuid.uuid4().hex[:6].upper()
    async with factory() as db:
        plant = Plant(code=f"PLT-{suffix}", name="Test Plant")
        db.add(plant)
        await db.flush()
        line = Line(code=f"LN-{suffix}", name="Test Line", plant_id=plant.id)
        db.add(line)
        await db.flush()
        machine = Machine(
            code=f"TSTM-{suffix}", name="Test Machine",
            machine_type="compressor", line_id=line.id, status="healthy",
        )
        db.add(machine)
        await db.commit()
        return machine.code


async def test_agents_summary_returns_shape(client):
    resp = await client.get("/api/v2/agents/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "by_agent" in body
    assert "totals" in body
    assert "total_calls" in body["totals"]


async def test_list_alerts_empty_ok(client):
    resp = await client.get("/api/v2/alerts", params={"machine_code": "NO-EXISTE"})
    assert resp.status_code == 200
    body = resp.json()
    assert {"total", "limit", "offset", "alerts"} <= set(body)


async def test_create_and_list_work_order(client):
    machine_code = await _seed_machine(client.session_factory)

    resp = await client.post("/api/v2/work-orders", json={
        "machine_code": machine_code,
        "title": "Inspección de rodamientos",
        "priority": "high",
        "estimated_cost": 1250.0,
    })
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["machine_code"] == machine_code
    assert created["priority"] == "high"

    resp = await client.get("/api/v2/work-orders", params={"machine_code": machine_code})
    assert resp.status_code == 200
    listed = resp.json()
    assert listed["total"] == 1
    assert listed["work_orders"][0]["title"] == "Inspección de rodamientos"


async def test_create_work_order_invalid_priority(client):
    resp = await client.post("/api/v2/work-orders", json={
        "machine_code": "IRRELEVANTE",
        "title": "x",
        "priority": "urgentísimo",
    })
    assert resp.status_code == 422
