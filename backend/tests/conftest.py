"""Fixtures pytest para tests de integración con PostgreSQL."""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

import app.infra.db.models  # noqa: F401  — registra modelos en Base.metadata
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://amia:amia_dev@localhost:5432/amia"
)


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """
    Sesión async por test, envuelta en una transacción externa que SIEMPRE
    se revierte al final — los tests no dejan datos en la BD compartida.
    Las tablas existen vía Alembic.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.connect() as conn:
        trans = await conn.begin()
        factory = async_sessionmaker(bind=conn, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            yield session
        await trans.rollback()
    await engine.dispose()
