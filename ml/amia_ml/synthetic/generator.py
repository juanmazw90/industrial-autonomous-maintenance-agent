"""
Synthetic industrial sensor data generator for AMIA.

Simulates 5 machines (pumps, motors, compressor) with realistic failure physics:
- Bearing wear: exponential vibration degradation over 100-300h
- Misalignment: sudden-onset vibration, stable then accelerating
- Electrical failure: current spikes with temperature correlation
- Overheating: temperature-driven, secondary current/vibration effects
- Cavitation: pressure fluctuations (pumps only)

Output columns per row:
  timestamp, machine_id, machine_type,
  vibration_rms, vibration_peak,
  temperature_bearing, temperature_motor,
  pressure_discharge (pumps/compressor),
  current_phase_a/b/c, speed_rpm,
  failure_mode, is_failure, rul_hours, degradation_fraction
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Machine configurations
# ---------------------------------------------------------------------------

MACHINE_CONFIGS: dict[str, dict[str, Any]] = {
    "PUMP-001": {
        "type": "centrifugal_pump",
        "nominal_speed": 1480.0,
        "nominal_current": 18.5,
        "nominal_vibration": 1.2,
        "nominal_temp_bearing": 55.0,
        "nominal_temp_motor": 48.0,
        "nominal_pressure": 6.2,
    },
    "PUMP-002": {
        "type": "centrifugal_pump",
        "nominal_speed": 1450.0,
        "nominal_current": 22.0,
        "nominal_vibration": 1.4,
        "nominal_temp_bearing": 58.0,
        "nominal_temp_motor": 51.0,
        "nominal_pressure": 7.1,
    },
    "MOTOR-001": {
        "type": "induction_motor",
        "nominal_speed": 1490.0,
        "nominal_current": 15.0,
        "nominal_vibration": 0.9,
        "nominal_temp_bearing": 52.0,
        "nominal_temp_motor": 62.0,
        "nominal_pressure": None,
    },
    "MOTOR-002": {
        "type": "induction_motor",
        "nominal_speed": 1485.0,
        "nominal_current": 28.0,
        "nominal_vibration": 1.1,
        "nominal_temp_bearing": 57.0,
        "nominal_temp_motor": 68.0,
        "nominal_pressure": None,
    },
    "COMP-001": {
        "type": "compressor",
        "nominal_speed": 2950.0,
        "nominal_current": 35.0,
        "nominal_vibration": 2.1,
        "nominal_temp_bearing": 65.0,
        "nominal_temp_motor": 72.0,
        "nominal_pressure": 8.2,
    },
}

FAILURE_MODES_BY_TYPE: dict[str, list[str]] = {
    "centrifugal_pump": ["bearing_wear", "misalignment", "electrical_failure", "overheating", "cavitation"],
    "induction_motor": ["bearing_wear", "misalignment", "electrical_failure", "overheating"],
    "compressor": ["bearing_wear", "misalignment", "electrical_failure", "overheating"],
}

# Typical degradation duration (min, max) in hours by failure mode
DEGRADATION_DURATION_HOURS: dict[str, tuple[int, int]] = {
    "bearing_wear": (120, 350),
    "misalignment": (40, 100),
    "electrical_failure": (20, 80),
    "overheating": (60, 150),
    "cavitation": (30, 90),
}

# Normal segment duration between failures (hours)
NORMAL_SEGMENT_HOURS = (600, 1800)


# ---------------------------------------------------------------------------
# Degradation physics helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray, k: float = 10.0, x0: float = 0.6) -> np.ndarray:
    """Sigmoid centered at x0 with steepness k, returns values in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _bearing_wear_multipliers(frac: np.ndarray) -> dict[str, np.ndarray]:
    """
    Bearing wear: vibration grows exponentially in the final 30% of life.
    RMS goes from 1x at frac=0 to ~4x at frac=1.
    """
    vib_mult = 1.0 + 3.5 * _sigmoid(frac, k=12, x0=0.65)
    temp_bearing_mult = 1.0 + 0.35 * frac
    temp_motor_mult = 1.0 + 0.12 * frac
    current_mult = 1.0 + 0.18 * frac
    speed_mult = 1.0 - 0.025 * frac
    return {
        "vib": vib_mult,
        "temp_bearing": temp_bearing_mult,
        "temp_motor": temp_motor_mult,
        "current": current_mult,
        "speed": speed_mult,
    }


def _misalignment_multipliers(frac: np.ndarray) -> dict[str, np.ndarray]:
    """
    Misalignment: sudden jump at onset (~frac=0.1), then slow progression.
    """
    onset = np.where(frac > 0.1, 1.0, 0.0)
    vib_mult = 1.0 + onset * (1.5 + 0.8 * frac)
    temp_bearing_mult = 1.0 + onset * 0.25 * frac
    temp_motor_mult = 1.0 + onset * 0.15 * frac
    current_mult = 1.0 + onset * (0.12 + 0.08 * frac)
    speed_mult = 1.0 - 0.01 * frac
    return {
        "vib": vib_mult,
        "temp_bearing": temp_bearing_mult,
        "temp_motor": temp_motor_mult,
        "current": current_mult,
        "speed": speed_mult,
    }


def _electrical_failure_multipliers(
    frac: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Electrical failure: random current spikes with frequency increasing over time.
    Spike probability grows from near-zero to ~30% per hour near failure.
    """
    spike_prob = 0.01 + 0.3 * frac**2
    spikes = rng.random(len(frac)) < spike_prob
    spike_mag = rng.uniform(1.8, 3.5, len(frac))

    current_mult = np.where(spikes, spike_mag, 1.0 + 0.08 * frac)
    # Temperature correlates with current spikes (thermal lag ~3h not modeled for simplicity)
    temp_bearing_mult = 1.0 + 0.05 * frac + 0.12 * (current_mult - 1.0)
    temp_motor_mult = 1.0 + 0.08 * frac + 0.18 * (current_mult - 1.0)
    vib_mult = 1.0 + 0.4 * frac + 0.1 * (current_mult - 1.0)
    speed_mult = 1.0 - 0.015 * frac
    return {
        "vib": vib_mult,
        "temp_bearing": temp_bearing_mult,
        "temp_motor": temp_motor_mult,
        "current": current_mult,
        "speed": speed_mult,
    }


def _overheating_multipliers(frac: np.ndarray) -> dict[str, np.ndarray]:
    """
    Overheating: temperature is the primary driver, current increases with load,
    motor may slow down due to thermal protection.
    """
    temp_bearing_mult = 1.0 + 0.9 * frac
    temp_motor_mult = 1.0 + 0.75 * frac
    current_mult = 1.0 + 0.28 * frac
    vib_mult = 1.0 + 0.22 * frac
    speed_mult = 1.0 - 0.06 * frac
    return {
        "vib": vib_mult,
        "temp_bearing": temp_bearing_mult,
        "temp_motor": temp_motor_mult,
        "current": current_mult,
        "speed": speed_mult,
    }


def _cavitation_multipliers(frac: np.ndarray) -> dict[str, np.ndarray]:
    """
    Cavitation (pumps only): pressure fluctuations are the hallmark.
    Also increases vibration and current.
    """
    pressure_std_mult = 1.0 + 4.0 * frac  # pressure noise multiplier
    vib_mult = 1.0 + 0.7 * frac
    temp_bearing_mult = 1.0 + 0.12 * frac
    temp_motor_mult = 1.0 + 0.08 * frac
    current_mult = 1.0 + 0.18 * frac
    speed_mult = 1.0 - 0.02 * frac
    return {
        "vib": vib_mult,
        "temp_bearing": temp_bearing_mult,
        "temp_motor": temp_motor_mult,
        "current": current_mult,
        "speed": speed_mult,
        "pressure_std": pressure_std_mult,
    }


MULTIPLIER_FN = {
    "bearing_wear": lambda frac, rng: _bearing_wear_multipliers(frac),
    "misalignment": lambda frac, rng: _misalignment_multipliers(frac),
    "electrical_failure": _electrical_failure_multipliers,
    "overheating": lambda frac, rng: _overheating_multipliers(frac),
    "cavitation": lambda frac, rng: _cavitation_multipliers(frac),
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class AMIADataGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Segment generators
    # ------------------------------------------------------------------

    def _normal_segment(
        self,
        machine_id: str,
        cfg: dict[str, Any],
        n_hours: int,
        start_hour: int,
        start_dt: datetime,
    ) -> pd.DataFrame:
        n = n_hours
        v0 = cfg["nominal_vibration"]
        t_b0 = cfg["nominal_temp_bearing"]
        t_m0 = cfg["nominal_temp_motor"]
        p0 = cfg["nominal_pressure"]
        i0 = cfg["nominal_current"]
        s0 = cfg["nominal_speed"]

        # Phase imbalance noise: normally balanced, slight random offsets
        i_a = self.rng.normal(i0, i0 * 0.04, n)
        i_b = self.rng.normal(i0, i0 * 0.04, n)
        i_c = self.rng.normal(i0, i0 * 0.04, n)

        vib_rms = np.clip(self.rng.normal(v0, v0 * 0.07, n), 0.1, None)
        vib_peak = vib_rms * self.rng.uniform(1.3, 1.7, n)

        rows: dict[str, Any] = {
            "timestamp": [start_dt + timedelta(hours=start_hour + i) for i in range(n)],
            "machine_id": machine_id,
            "machine_type": cfg["type"],
            "vibration_rms": vib_rms,
            "vibration_peak": vib_peak,
            "temperature_bearing": np.clip(self.rng.normal(t_b0, t_b0 * 0.04, n), 20.0, None),
            "temperature_motor": np.clip(self.rng.normal(t_m0, t_m0 * 0.04, n), 20.0, None),
            "pressure_discharge": (
                np.clip(self.rng.normal(p0, p0 * 0.035, n), 0.5, None)
                if p0 is not None
                else np.full(n, np.nan)
            ),
            "current_phase_a": np.clip(i_a, 0.5, None),
            "current_phase_b": np.clip(i_b, 0.5, None),
            "current_phase_c": np.clip(i_c, 0.5, None),
            "speed_rpm": np.clip(self.rng.normal(s0, s0 * 0.003, n), 1.0, None),
            "failure_mode": "normal",
            "is_failure": False,
            "degradation_fraction": 0.0,
        }
        return pd.DataFrame(rows)

    def _degradation_segment(
        self,
        machine_id: str,
        cfg: dict[str, Any],
        failure_mode: str,
        n_hours: int,
        start_hour: int,
        start_dt: datetime,
    ) -> pd.DataFrame:
        n = n_hours
        v0 = cfg["nominal_vibration"]
        t_b0 = cfg["nominal_temp_bearing"]
        t_m0 = cfg["nominal_temp_motor"]
        p0 = cfg["nominal_pressure"]
        i0 = cfg["nominal_current"]
        s0 = cfg["nominal_speed"]

        frac = np.linspace(0.0, 1.0, n)
        mults = MULTIPLIER_FN[failure_mode](frac, self.rng)

        vib_base = v0 * mults["vib"]
        vib_rms = np.clip(
            vib_base * self.rng.normal(1.0, 0.06, n),
            0.1,
            None,
        )
        vib_peak = vib_rms * self.rng.uniform(1.3, 1.8, n)

        temp_b = np.clip(
            t_b0 * mults["temp_bearing"] * self.rng.normal(1.0, 0.035, n),
            20.0,
            None,
        )
        temp_m = np.clip(
            t_m0 * mults["temp_motor"] * self.rng.normal(1.0, 0.035, n),
            20.0,
            None,
        )

        # Phase imbalance grows with electrical failure
        if failure_mode == "electrical_failure":
            imbalance = 0.04 + 0.25 * frac
        else:
            imbalance = np.full(n, 0.04)

        current_base = i0 * mults["current"]
        i_a = np.clip(current_base * self.rng.normal(1.0, imbalance, n), 0.5, None)
        i_b = np.clip(current_base * self.rng.normal(1.0, imbalance, n), 0.5, None)
        i_c = np.clip(current_base * self.rng.normal(1.0, imbalance, n), 0.5, None)

        speed = np.clip(
            s0 * mults["speed"] * self.rng.normal(1.0, 0.003, n),
            1.0,
            None,
        )

        if p0 is not None:
            p_std_mult = mults.get("pressure_std", np.ones(n))
            pressure = np.clip(
                self.rng.normal(p0, p0 * 0.035 * p_std_mult, n),
                0.5,
                None,
            )
        else:
            pressure = np.full(n, np.nan)

        # Mark the final hour as the failure event
        is_failure = np.zeros(n, dtype=bool)
        is_failure[-1] = True
        failure_modes = np.full(n, failure_mode, dtype=object)
        failure_modes[-1] = "failure"

        return pd.DataFrame({
            "timestamp": [start_dt + timedelta(hours=start_hour + i) for i in range(n)],
            "machine_id": machine_id,
            "machine_type": cfg["type"],
            "vibration_rms": vib_rms,
            "vibration_peak": vib_peak,
            "temperature_bearing": temp_b,
            "temperature_motor": temp_m,
            "pressure_discharge": pressure,
            "current_phase_a": i_a,
            "current_phase_b": i_b,
            "current_phase_c": i_c,
            "speed_rpm": speed,
            "failure_mode": failure_modes,
            "is_failure": is_failure,
            "degradation_fraction": frac,
        })

    # ------------------------------------------------------------------
    # Machine history
    # ------------------------------------------------------------------

    def generate_machine_history(
        self,
        machine_id: str,
        cfg: dict[str, Any],
        total_hours: int,
        start_dt: datetime,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        segments: list[pd.DataFrame] = []
        failure_events: list[dict[str, Any]] = []
        current_hour = 0
        available_modes = FAILURE_MODES_BY_TYPE[cfg["type"]]

        while current_hour < total_hours:
            # Normal segment
            normal_dur = int(self.rng.integers(*NORMAL_SEGMENT_HOURS))
            normal_dur = min(normal_dur, total_hours - current_hour)
            segments.append(
                self._normal_segment(machine_id, cfg, normal_dur, current_hour, start_dt)
            )
            current_hour += normal_dur

            if current_hour >= total_hours:
                break

            # Degradation segment
            mode = self.rng.choice(available_modes)
            deg_min, deg_max = DEGRADATION_DURATION_HOURS[mode]
            deg_dur = int(self.rng.integers(deg_min, deg_max + 1))
            deg_dur = min(deg_dur, total_hours - current_hour)

            segments.append(
                self._degradation_segment(machine_id, cfg, mode, deg_dur, current_hour, start_dt)
            )
            failure_events.append({
                "machine_id": machine_id,
                "failure_mode": mode,
                "failure_hour": current_hour + deg_dur - 1,
                "degradation_start_hour": current_hour,
                "degradation_duration_hours": deg_dur,
                "failure_timestamp": (start_dt + timedelta(hours=current_hour + deg_dur - 1)).isoformat(),
                "degradation_start_timestamp": (start_dt + timedelta(hours=current_hour)).isoformat(),
            })
            current_hour += deg_dur

        df = pd.concat(segments, ignore_index=True)
        df = self._add_rul(df, failure_events)
        return df, failure_events

    # ------------------------------------------------------------------
    # RUL calculation
    # ------------------------------------------------------------------

    def _add_rul(
        self,
        df: pd.DataFrame,
        failure_events: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        For each row compute hours until next failure.
        Rows with no upcoming failure within MAX_RUL_HORIZON are set to NaN
        to avoid mixing long-range healthy readings with actual RUL predictions.
        """
        MAX_RUL_HORIZON = 500  # only label if failure is within this many hours

        rul = np.full(len(df), np.nan)
        failure_hours = sorted(e["failure_hour"] for e in failure_events)

        for idx, row_hour in enumerate(range(len(df))):
            upcoming = [fh for fh in failure_hours if fh >= row_hour]
            if upcoming:
                hours_to_failure = upcoming[0] - row_hour
                if hours_to_failure <= MAX_RUL_HORIZON:
                    rul[idx] = hours_to_failure

        df["rul_hours"] = rul
        return df

    # ------------------------------------------------------------------
    # Full dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        total_days: int = 365,
        start_dt: datetime | None = None,
        machines: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if start_dt is None:
            start_dt = datetime(2024, 1, 1, 0, 0, 0)
        if machines is None:
            machines = MACHINE_CONFIGS

        total_hours = total_days * 24
        all_dfs: list[pd.DataFrame] = []
        all_events: list[dict[str, Any]] = []

        for machine_id, cfg in machines.items():
            print(f"  Generating {machine_id} ({cfg['type']}) — {total_days} days...")
            df_machine, events = self.generate_machine_history(
                machine_id, cfg, total_hours, start_dt
            )
            all_dfs.append(df_machine)
            all_events.extend(events)

        dataset = pd.concat(all_dfs, ignore_index=True)
        dataset = dataset.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        return dataset, all_events


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AMIA synthetic industrial dataset")
    parser.add_argument("--days", type=int, default=365, help="Days of data to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.days} days of synthetic industrial data (seed={args.seed})...")
    gen = AMIADataGenerator(seed=args.seed)
    df, events = gen.generate_dataset(total_days=args.days)

    # Save dataset
    parquet_path = output_dir / "sensor_readings.parquet"
    csv_path = output_dir / "sensor_readings.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved:")
    print(f"  {parquet_path}  ({parquet_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {csv_path}  ({csv_path.stat().st_size / 1024:.0f} KB)")

    # Save failure events
    events_path = output_dir / "failure_events.json"
    events_path.write_text(json.dumps(events, indent=2))
    print(f"  {events_path}  ({len(events)} events)")

    # Save machine configs
    configs_path = output_dir / "machine_configs.json"
    configs_path.write_text(json.dumps(MACHINE_CONFIGS, indent=2))

    # Stats summary
    print(f"\nDataset summary:")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Machines:         {df['machine_id'].nunique()}")
    print(f"  Failure events:   {len(events)}")
    print(f"  Failure modes:    {df[df['failure_mode'] != 'normal']['failure_mode'].value_counts().to_dict()}")
    print(f"  Date range:       {df['timestamp'].min()} → {df['timestamp'].max()}")

    labeled_mask = df["rul_hours"].notna()
    print(f"  Rows with RUL:    {labeled_mask.sum():,} ({100*labeled_mask.mean():.1f}%)")

    stats = {
        "total_rows": len(df),
        "machines": df["machine_id"].nunique(),
        "failure_events": len(events),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "failure_mode_counts": df["failure_mode"].value_counts().to_dict(),
        "rows_with_rul": int(labeled_mask.sum()),
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    print("\nDone. Run 'uv run scripts/visualize_data.py' to generate EDA plots.")


if __name__ == "__main__":
    main()
