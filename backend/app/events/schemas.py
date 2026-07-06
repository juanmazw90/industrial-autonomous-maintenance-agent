"""Schemas Pydantic de todos los eventos del sistema (SDD sección 5.3)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    type: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunFinishedEvent(BaseEvent):
    type: Literal["agent_run.finished"] = "agent_run.finished"


class ToolCallFinishedEvent(BaseEvent):
    type: Literal["tool_call.finished"] = "tool_call.finished"


class AlertCreatedEvent(BaseEvent):
    type: Literal["alert.created"] = "alert.created"


class WorkOrderCreatedEvent(BaseEvent):
    type: Literal["wo.created"] = "wo.created"


class PredictionCreatedEvent(BaseEvent):
    type: Literal["prediction.created"] = "prediction.created"


class ServiceHealthEvent(BaseEvent):
    type: Literal["service.health"] = "service.health"
