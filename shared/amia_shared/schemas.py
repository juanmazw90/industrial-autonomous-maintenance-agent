from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MachineType(StrEnum):
    CENTRIFUGAL_PUMP = "centrifugal_pump"
    INDUCTION_MOTOR = "induction_motor"
    COMPRESSOR = "compressor"


class FailureMode(StrEnum):
    NORMAL = "normal"
    BEARING_WEAR = "bearing_wear"
    MISALIGNMENT = "misalignment"
    ELECTRICAL_FAILURE = "electrical_failure"
    OVERHEATING = "overheating"
    CAVITATION = "cavitation"
    FAILURE = "failure"


class MachineConfig(BaseModel):
    machine_id: str
    machine_type: MachineType
    nominal_speed_rpm: float
    nominal_current_a: float
    nominal_vibration_rms: float
    nominal_temperature_c: float
    nominal_pressure_bar: float | None = None
    location: str = "plant_floor"


class SensorReading(BaseModel):
    timestamp: datetime
    machine_id: str
    machine_type: MachineType
    vibration_rms: float = Field(description="Vibration RMS in mm/s")
    vibration_peak: float = Field(description="Vibration peak in mm/s")
    temperature_bearing: float = Field(description="Bearing temperature in °C")
    temperature_motor: float = Field(description="Motor/housing temperature in °C")
    pressure_discharge: float | None = Field(None, description="Discharge pressure in bar")
    current_phase_a: float = Field(description="Phase A current in A")
    current_phase_b: float = Field(description="Phase B current in A")
    current_phase_c: float = Field(description="Phase C current in A")
    speed_rpm: float = Field(description="Rotational speed in RPM")
    failure_mode: FailureMode = FailureMode.NORMAL
    is_failure: bool = False
    rul_hours: float | None = Field(None, description="Remaining useful life in hours")
    degradation_fraction: float = Field(0.0, ge=0.0, le=1.0)


class FailureEvent(BaseModel):
    machine_id: str
    failure_mode: FailureMode
    failure_timestamp: datetime
    degradation_start_timestamp: datetime
    degradation_duration_hours: int
