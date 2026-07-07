import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InputQuery(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class FailurePredictionResponse(BaseModel):
    machine_id: str
    failure_probability: float
    risk_score: float
    alert_level: Literal["green", "yellow", "red"]
    is_high_risk: bool
    threshold_used: float
    as_of_timestamp: str


class IncomingSensorReading(BaseModel):
    """Lectura de sensores recibida del simulador o de un sistema SCADA real."""
    timestamp: datetime
    machine_id: str
    machine_type: str
    vibration_rms: float
    vibration_peak: float
    temperature_bearing: float
    temperature_motor: float
    pressure_discharge: float | None = None
    current_phase_a: float
    current_phase_b: float
    current_phase_c: float
    speed_rpm: float

