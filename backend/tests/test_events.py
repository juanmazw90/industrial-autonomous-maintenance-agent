"""Tests Etapa 0.4 — event bus y schemas."""
from __future__ import annotations

import json

from app.events.schemas import (
    AgentRunFinishedEvent,
    AlertCreatedEvent,
    BaseEvent,
)


def test_base_event_defaults():
    e = BaseEvent(type="test.event")
    assert e.correlation_id
    assert e.ts
    assert e.payload == {}


def test_agent_run_finished_event():
    e = AgentRunFinishedEvent(payload={"agent": "supervisor", "latency_ms": 820})
    assert e.type == "agent_run.finished"
    data = json.loads(e.model_dump_json())
    assert data["payload"]["latency_ms"] == 820


def test_alert_created_event():
    e = AlertCreatedEvent(
        payload={"machine_id": "MCH-02", "severity": "critical"},
        correlation_id="test-corr-id",
    )
    assert e.type == "alert.created"
    assert e.correlation_id == "test-corr-id"
