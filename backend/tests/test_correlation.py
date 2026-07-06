"""Tests Etapa 0.3 — logging estructurado + correlation_id."""
from __future__ import annotations

import pytest
import structlog

from app.observability.logging import configure_logging


def test_configure_logging_runs():
    configure_logging(debug=False)
    log = structlog.get_logger("test")
    log.info("test_log_entry", key="value")


def test_structlog_contextvars():
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id="abc-123")
    ctx = structlog.contextvars.get_contextvars()
    assert ctx["correlation_id"] == "abc-123"
    structlog.contextvars.clear_contextvars()
