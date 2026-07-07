"""Configuración centralizada con pydantic-settings. Única fuente de verdad."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/infra/settings.py → raíz del repo
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # env_file absoluto: el comportamiento no depende del cwd desde el que se arranque
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    llm_supervisor_model: str = "claude-haiku-4-5-20251001"
    llm_synthesizer_model: str = "claude-sonnet-4-6"

    # Database (asyncpg driver para SQLAlchemy async)
    database_url: str = "postgresql+asyncpg://amia:amia_dev@localhost:5432/amia"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # MLflow / datos (rutas absolutas: evita depender del cwd)
    mlflow_tracking_uri: str = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    data_path: str = str(REPO_ROOT / "data/synthetic/sensor_readings.parquet")

    # Langfuse (opcional)
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Rate limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # API
    app_version: str = "2.0.0"
    debug: bool = False
    cors_origins: str = "http://localhost:3000"  # separadas por coma
    log_file_path: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
