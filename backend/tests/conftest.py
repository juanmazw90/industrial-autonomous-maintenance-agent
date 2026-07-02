"""Fixtures pytest para tests de integración con PostgreSQL."""
from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.infra.db.models  # noqa: F401  — registra modelos en Base.metadata

TEST_DATABASE_URL = "postgresql+asyncpg://amia:amia_dev@localhost:5432/amia"


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    """Sesión async por test. Las tablas existen vía Alembic. Los IDs son UUID únicos."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()
