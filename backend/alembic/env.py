"""Alembic env.py — async SQLAlchemy 2.0."""
from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Asegura que el package `app` sea importable desde backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.infra.db.models  # noqa: F401, E402  — registra todos los modelos en Base.metadata
from app.infra.db.base import Base  # noqa: E402
from app.infra.settings import settings  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
