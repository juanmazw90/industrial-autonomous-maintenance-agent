"""
sensor_simulator.py — Reproduce lecturas históricas de sensores hacia el backend.

Replay el dataset sintético a velocidad acelerada para demostrar el dashboard
en tiempo real: máquinas en verde, transición a amarillo, spike a rojo.

Uso:
    uv run python scripts/sensor_simulator.py
    uv run python scripts/sensor_simulator.py --start-date 2024-10-01 --interval 0.2
    uv run python scripts/sensor_simulator.py --help

Variables de entorno:
    BACKEND_URL  (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data/synthetic/sensor_readings.parquet"
DEFAULT_BACKEND = "http://localhost:8000"
DEFAULT_START   = "2024-10-01"
DEFAULT_INTERVAL = 0.5   # segundos reales entre ticks (cada tick = 1h simulada)

ALERT_EMOJI  = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
ALERT_COLORS = {
    "green":  "\033[92m",   # verde
    "yellow": "\033[93m",   # amarillo
    "red":    "\033[91m",   # rojo
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _color(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _format_prediction(pred: dict) -> str:
    mid   = pred.get("machine_id", "?")
    prob  = pred.get("failure_probability", 0.0)
    level = pred.get("alert_level", "green")
    emoji = ALERT_EMOJI.get(level, "⚪")
    color = ALERT_COLORS.get(level, "")
    return _color(f"{emoji} {mid}: {prob:.1%}", color)


def _print_tick(sim_ts: datetime, results: list[dict], alerts: list[str]) -> None:
    ts_str = sim_ts.strftime("%Y-%m-%d %H:%M")
    row = "  ".join(_format_prediction(r) for r in results)
    print(f"\r{BOLD}[{ts_str}]{RESET}  {row}", end="", flush=True)
    if alerts:
        print()  # nueva línea para las alertas
        for a in alerts:
            print(f"  {BOLD}⚠️  ALERTA:{RESET} {a}")


def run(
    backend_url: str,
    start_date: str,
    interval: float,
) -> None:
    print(f"\n{BOLD}AMIA Sensor Simulator{RESET}")
    print(f"Backend:    {backend_url}")
    print(f"Datos:      {DATA_PATH}")
    print(f"Desde:      {start_date}")
    print(f"Intervalo:  {interval}s por hora simulada")

    # Verificar que el backend está disponible y el predictor listo
    try:
        resp = httpx.get(f"{backend_url}/health", timeout=5)
        health = resp.json()
        if not health.get("predictor_ready"):
            print("\n❌ El predictor no está listo. ¿Arrancaste el backend?")
            print("   Ejecuta: uv run uvicorn app.main:app --app-dir backend --port 8000")
            sys.exit(1)
        print("Backend:    ✅ OK (predictor listo)\n")
    except httpx.ConnectError:
        print(f"\n❌ No se puede conectar a {backend_url}")
        print("   Ejecuta: uv run uvicorn app.main:app --app-dir backend --port 8000")
        sys.exit(1)

    # Cargar datos
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    cut = pd.Timestamp(start_date)
    df  = df[df["timestamp"] >= cut].reset_index(drop=True)
    if df.empty:
        print(f"❌ No hay datos desde {start_date}")
        sys.exit(1)

    machines   = sorted(df["machine_id"].unique())
    timestamps = sorted(df["timestamp"].unique())
    total_h    = len(timestamps)

    print(f"Máquinas:   {', '.join(machines)}")
    print(f"Período:    {timestamps[0]} → {timestamps[-1]}")
    print(f"Ticks:      {total_h:,} horas simuladas\n")
    print("Presiona Ctrl+C para detener.\n")

    prev_levels: dict[str, str] = {}
    high_risk_count = 0

    with httpx.Client(base_url=backend_url, timeout=10) as client:
        try:
            for ts in timestamps:
                tick_df = df[df["timestamp"] == ts]
                results: list[dict] = []
                alerts: list[str]   = []

                for _, row in tick_df.iterrows():
                    payload = {
                        "timestamp":           row["timestamp"].isoformat(),
                        "machine_id":          row["machine_id"],
                        "machine_type":        row["machine_type"],
                        "vibration_rms":       float(row["vibration_rms"]),
                        "vibration_peak":      float(row["vibration_peak"]),
                        "temperature_bearing": float(row["temperature_bearing"]),
                        "temperature_motor":   float(row["temperature_motor"]),
                        "pressure_discharge":  (float(row["pressure_discharge"])
                                                if pd.notna(row.get("pressure_discharge")) else None),
                        "current_phase_a":     float(row["current_phase_a"]),
                        "current_phase_b":     float(row["current_phase_b"]),
                        "current_phase_c":     float(row["current_phase_c"]),
                        "speed_rpm":           float(row["speed_rpm"]),
                    }
                    try:
                        resp = client.post("/sensors/reading", json=payload)
                        resp.raise_for_status()
                        pred = resp.json()
                    except Exception as e:
                        pred = {
                            "machine_id":          row["machine_id"],
                            "failure_probability": 0.0,
                            "alert_level":         "green",
                            "error":               str(e),
                        }

                    results.append(pred)

                    # Detectar cambios de nivel de alerta
                    mid   = pred.get("machine_id", "")
                    level = pred.get("alert_level", "green")
                    prev  = prev_levels.get(mid, "green")
                    if level != prev:
                        prob = pred.get("failure_probability", 0.0)
                        emoji = ALERT_EMOJI.get(level, "⚪")
                        alerts.append(
                            f"{mid}: {ALERT_EMOJI.get(prev, '⚪')} → {emoji}  "
                            f"({prob:.1%} probabilidad de fallo)"
                        )
                        prev_levels[mid] = level
                        if level == "red":
                            high_risk_count += 1

                sim_ts = pd.Timestamp(ts).to_pydatetime()
                _print_tick(sim_ts, results, alerts)
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\n{BOLD}Simulación detenida.{RESET}")

    print(f"\n\n{'='*60}")
    print(f"{BOLD}Resumen de la simulación{RESET}")
    print(f"  Período simulado:    {timestamps[0]} → {timestamps[-1]}")
    print(f"  Ticks procesados:    {total_h:,}")
    print(f"  Alarmas rojas:       {high_risk_count}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador de sensores AMIA — replica lecturas históricas al backend."
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START,
        help=f"Fecha de inicio del replay (YYYY-MM-DD). Default: {DEFAULT_START}"
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL,
        help=f"Segundos entre ticks (cada tick = 1h simulada). Default: {DEFAULT_INTERVAL}"
    )
    parser.add_argument(
        "--backend-url", default=DEFAULT_BACKEND,
        help=f"URL del backend AMIA. Default: {DEFAULT_BACKEND}"
    )
    args = parser.parse_args()
    run(args.backend_url, args.start_date, args.interval)


if __name__ == "__main__":
    main()
