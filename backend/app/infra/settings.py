"""Configuración centralizada con pydantic-settings. Única fuente de verdad."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = ""

    # Database (asyncpg driver para SQLAlchemy async)
    database_url: str = "postgresql+asyncpg://amia:amia_dev@localhost:5432/amia"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Langfuse (opcional)
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Rate limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # API
    app_version: str = "2.0.0"
    debug: bool = False


settings = Settings()
