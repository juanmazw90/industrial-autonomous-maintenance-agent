"""
seed_demo.py — Escenario de demo determinista para AMIA Platform v2.

Machine codes match the ML layer (data/synthetic/sensor_readings.parquet):
  COMP-001, MOTOR-001, MOTOR-002, PUMP-001, PUMP-002

Genera el dataset de demostración descrito en SDD sección 0.5:
  - 5 demo_users (uno por rol)
  - Plant Alpha > 2 lines > 5 machines
  - PUMP-001 en estado crítico: prob. fallo 98%, RUL 18h
  - Timeline completo de PUMP-001: anomalía → predicción → RCA → económico → WO
  - 3 alertas abiertas (1 critical, 1 high, 1 medium)
  - 2 work orders (1 open, 1 in_progress)
  - 10 agent_runs con tool_calls de ejemplo
  - 5 rag_queries de ejemplo

Uso:
  uv run python scripts/seed_demo.py           # siembra si no existe (idempotente por code único)
  uv run python scripts/seed_demo.py --reset   # limpia y re-siembra
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Añadir backend al sys.path para importar app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db.models import (
    AgentRun,
    Alert,
    AuditLog,
    DemoUser,
    Line,
    Machine,
    ModelPrediction,
    Plant,
    PlatformConfig,
    RagQuery,
    TimelineEvent,
    ToolCall,
    WorkOrder,
)
from app.infra.settings import settings

engine = create_async_engine(settings.database_url, echo=False)
Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

NOW = datetime.now(timezone.utc)


def _dt(hours_ago: float = 0) -> datetime:
    return NOW - timedelta(hours=hours_ago)


# ── IDs fijos para el escenario determinista ──────────────────────────────────
# Machine codes MUST match ML layer (data/synthetic/sensor_readings.parquet)

PLANT_ID  = "00000000-0000-0000-0000-000000000001"
LINE1_ID  = "00000000-0000-0000-0000-000000000010"
LINE2_ID  = "00000000-0000-0000-0000-000000000011"
MCH = {
    "COMP-001":  "00000000-0000-0000-0000-000000000101",
    "PUMP-001":  "00000000-0000-0000-0000-000000000102",  # ← crítico (98% failure)
    "MOTOR-001": "00000000-0000-0000-0000-000000000103",
    "MOTOR-002": "00000000-0000-0000-0000-000000000104",
    "PUMP-002":  "00000000-0000-0000-0000-000000000105",
}
USERS = {
    "operator":            "00000000-0000-0000-0001-000000000001",
    "supervisor":          "00000000-0000-0000-0001-000000000002",
    "maintenance_manager": "00000000-0000-0000-0001-000000000003",
    "plant_director":      "00000000-0000-0000-0001-000000000004",
    "ai_engineer":         "00000000-0000-0000-0001-000000000005",
}
PRED_ID     = "00000000-0000-0000-0002-000000000001"
ALERT_C_ID  = "00000000-0000-0000-0003-000000000001"
ALERT_H_ID  = "00000000-0000-0000-0003-000000000002"
ALERT_M_ID  = "00000000-0000-0000-0003-000000000003"
WO_OPEN_ID  = "00000000-0000-0000-0004-000000000001"
WO_PROG_ID  = "00000000-0000-0000-0004-000000000002"
RUN_BASE_ID = "00000000-0000-0000-0005-{:012d}"


async def _reset(session: AsyncSession) -> None:
    print("Limpiando datos de demo...")
    for model in [
        RagQuery, ToolCall, AgentRun,
        AuditLog, TimelineEvent,
        WorkOrder, Alert, ModelPrediction,
        Machine, Line, Plant,
        DemoUser, PlatformConfig,
    ]:
        await session.execute(delete(model))
    await session.commit()
    print("   done")


async def _seed(session: AsyncSession) -> None:
    print("Sembrando escenario de demo AMIA v2...")

    # ── Demo users ─────────────────────────────────────────────────────────────
    users = [
        DemoUser(id=uid, name=_role_name(role), role=role)
        for role, uid in USERS.items()
    ]
    session.add_all(users)
    await session.flush()
    print(f"   OK {len(users)} demo_users")

    # ── Plant / Lines / Machines ───────────────────────────────────────────────
    plant = Plant(id=PLANT_ID, code="PLANT-ALPHA", name="Plant Alpha")
    session.add(plant)
    await session.flush()

    line1 = Line(id=LINE1_ID, code="LINE-01", name="Production Line 1", plant_id=PLANT_ID)
    line2 = Line(id=LINE2_ID, code="LINE-02", name="Production Line 2", plant_id=PLANT_ID)
    session.add_all([line1, line2])
    await session.flush()

    machines_data = [
        ("COMP-001",  "Compressor 001",       "compressor",       LINE1_ID, "healthy"),
        ("PUMP-001",  "Centrifugal Pump 001",  "centrifugal_pump", LINE1_ID, "critical"),
        ("MOTOR-001", "Induction Motor 001",   "induction_motor",  LINE2_ID, "warning"),
        ("MOTOR-002", "Induction Motor 002",   "induction_motor",  LINE2_ID, "healthy"),
        ("PUMP-002",  "Centrifugal Pump 002",  "centrifugal_pump", LINE2_ID, "healthy"),
    ]
    for code, name, mtype, line_id, status in machines_data:
        session.add(Machine(
            id=MCH[code], code=code, name=name,
            machine_type=mtype, line_id=line_id, status=status,
        ))
    await session.flush()
    print("   OK Plant Alpha, 2 lines, 5 machines (codes aligned with ML layer)")

    # ── Platform config (valores por defecto) ─────────────────────────────────
    configs = {
        "thresholds.failure":  {"value": 0.85},
        "thresholds.economic": {"value": 5000},
        "rag.top_k":           {"value": 5},
        "rag.min_similarity":  {"value": 0.65},
        "llm.temperature":     {"value": 0.1},
        "llm.max_tokens":      {"value": 4096},
        "llm.model":           {"value": "claude-sonnet-4-6"},
        "llm.price_input":     {"value": 0.003},
        "llm.price_output":    {"value": 0.015},
    }
    for key, val in configs.items():
        session.add(PlatformConfig(key=key, value=val, updated_by="seed_demo"))
    await session.flush()
    print("   OK platform_config defaults")

    # ── Predicción crítica PUMP-001 ───────────────────────────────────────────
    pred = ModelPrediction(
        id=PRED_ID,
        machine_id=MCH["PUMP-001"],
        model_name="amia-failure-model",
        model_version="1",
        prediction={"failure_probability": 0.98, "risk_score": 0.96, "alert_level": "red"},
        probability=0.98,
        shap_top_features=[
            {"feature": "vibration_rms",         "shap_value": 0.42, "direction": "positive"},
            {"feature": "temperature_bearing",    "shap_value": 0.31, "direction": "positive"},
            {"feature": "vibration_peak",         "shap_value": 0.18, "direction": "positive"},
            {"feature": "current_phase_a",        "shap_value": 0.09, "direction": "positive"},
            {"feature": "vibration_rms_slope_8h", "shap_value": 0.07, "direction": "positive"},
            {"feature": "temp_per_rpm",           "shap_value": -0.04, "direction": "negative"},
            {"feature": "pressure_discharge",     "shap_value": 0.03, "direction": "positive"},
            {"feature": "speed_rpm_mean_24h",     "shap_value": -0.02, "direction": "negative"},
        ],
        created_at=_dt(2),
    )
    session.add(pred)
    await session.flush()

    # ── Alertas ───────────────────────────────────────────────────────────────
    alert_critical = Alert(
        id=ALERT_C_ID,
        machine_id=MCH["PUMP-001"],
        severity="critical",
        source="failure_model",
        title="PUMP-001 — Failure probability 98% (RUL 18h)",
        description=(
            "Centrifugal Pump 001 shows critical bearing wear pattern. "
            "Model predicts failure within 18 hours. Immediate intervention required."
        ),
        status="acknowledged",
        acknowledged_by=USERS["supervisor"],
        prediction_id=PRED_ID,
        work_order_id=WO_OPEN_ID,
        created_at=_dt(1.8),
        updated_at=_dt(1.5),
    )
    alert_high = Alert(
        id=ALERT_H_ID,
        machine_id=MCH["MOTOR-001"],
        severity="high",
        source="sensor_threshold",
        title="MOTOR-001 — High vibration detected",
        description="Induction Motor 001 vibration_rms exceeded 12 mm/s threshold.",
        status="assigned",
        acknowledged_by=USERS["operator"],
        assigned_to=USERS["maintenance_manager"],
        created_at=_dt(4),
        updated_at=_dt(3),
    )
    alert_medium = Alert(
        id=ALERT_M_ID,
        machine_id=MCH["COMP-001"],
        severity="medium",
        source="rul_model",
        title="COMP-001 — RUL below 300h threshold",
        description="Compressor 001 estimated remaining useful life: 234h.",
        status="new",
        created_at=_dt(6),
    )
    session.add_all([alert_critical, alert_high, alert_medium])
    await session.flush()
    print("   OK 3 alerts (critical/high/medium)")

    # ── Work Orders ───────────────────────────────────────────────────────────
    wo_open = WorkOrder(
        id=WO_OPEN_ID,
        machine_id=MCH["PUMP-001"],
        alert_id=ALERT_C_ID,
        title="Emergency bearing replacement — PUMP-001",
        description=(
            "Replace bearing assembly on Centrifugal Pump 001. "
            "Predicted failure mode: bearing_wear (confidence 0.87). "
            "Economic risk: $12,400 downtime cost."
        ),
        priority="critical",
        status="open",
        estimated_cost=3200.0,
        created_at=_dt(1.5),
        updated_at=_dt(1.5),
    )
    wo_progress = WorkOrder(
        id=WO_PROG_ID,
        machine_id=MCH["MOTOR-001"],
        alert_id=ALERT_H_ID,
        title="Vibration inspection and balancing — MOTOR-001",
        description="Inspect and balance rotor on Induction Motor 001.",
        priority="high",
        status="in_progress",
        assigned_to=USERS["maintenance_manager"],
        estimated_cost=800.0,
        created_at=_dt(3.5),
        updated_at=_dt(1),
    )
    session.add_all([wo_open, wo_progress])
    await session.flush()
    print("   OK 2 work orders (open, in_progress)")

    # ── Timeline PUMP-001 (escenario completo) ────────────────────────────────
    timeline = [
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(2.5),
            kind="sensor_anomaly",
            title="High vibration detected (vibration_rms: 14.2 mm/s)",
            payload={"sensor": "vibration_rms", "value": 14.2, "threshold": 10.0},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(2.0),
            kind="prediction",
            title="Failure model v1 → 98% probability",
            payload={"model": "amia-failure-model", "probability": 0.98, "prediction_id": PRED_ID},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(1.9),
            kind="agent_decision",
            title="Supervisor routed → sensor_analyst",
            payload={"from": "supervisor", "to": "sensor_analyst",
                     "reason": "sensor anomaly + high failure prob"},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(1.8),
            kind="rca",
            title="RCA: Bearing Wear (confidence 0.87)",
            payload={"failure_mode": "bearing_wear", "confidence": 0.87,
                     "model": "amia-rca-model"},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(1.75),
            kind="economic",
            title="Economic impact: $12,400 downtime risk",
            payload={"downtime_cost_usd": 12400, "repair_cost_usd": 3200,
                     "total_risk_usd": 15600},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(1.5),
            kind="alert_created",
            title="Alert CRITICAL created — PUMP-001",
            payload={"alert_id": ALERT_C_ID, "severity": "critical"},
            correlation_id="demo-corr-001",
        ),
        TimelineEvent(
            machine_id=MCH["PUMP-001"], ts=_dt(1.5),
            kind="wo_created",
            title="WO created: Emergency bearing replacement",
            payload={"work_order_id": WO_OPEN_ID, "priority": "critical",
                     "estimated_cost": 3200},
            correlation_id="demo-corr-001",
        ),
    ]
    session.add_all(timeline)
    await session.flush()
    print("   OK 7 timeline events (PUMP-001 full scenario)")

    # ── Agent runs + tool calls ───────────────────────────────────────────────
    runs = []
    all_tools = []
    agent_defs = [
        ("supervisor",        "claude-haiku-4-5-20251001", 820,  150,  80,  0.001),
        ("sensor_analyst",    "claude-haiku-4-5-20251001", 2100, 900,  450, 0.008),
        ("rul_analyst",       "claude-haiku-4-5-20251001", 1400, 600,  300, 0.005),
        ("economic_analyst",  "claude-haiku-4-5-20251001", 1800, 700,  350, 0.006),
        ("work_order_creator","claude-haiku-4-5-20251001", 950,  400,  200, 0.004),
        ("synthesizer",       "claude-haiku-4-5-20251001", 1200, 500,  800, 0.007),
        ("doc_expert",        "claude-haiku-4-5-20251001", 1600, 800,  600, 0.009),
        ("supervisor",        "claude-haiku-4-5-20251001", 710,  120,  70,  0.001),
        ("sensor_analyst",    "claude-haiku-4-5-20251001", 1900, 750,  400, 0.007),
        ("synthesizer",       "claude-haiku-4-5-20251001", 1100, 450,  700, 0.006),
    ]
    for i, (agent, model, lat, inp_tok, out_tok, cost) in enumerate(agent_defs):
        run_id = RUN_BASE_ID.format(i + 1)
        run = AgentRun(
            id=run_id,
            session_id="demo-session-001" if i < 7 else "demo-session-002",
            agent_name=agent,
            started_at=_dt(2 - i * 0.05),
            finished_at=_dt(2 - i * 0.05 - lat / 3_600_000),
            latency_ms=lat,
            input_summary=f"Demo input for {agent} run {i + 1}",
            output_summary=f"Demo output for {agent} run {i + 1}",
            reasoning_summary=(
                f"{agent.replace('_', ' ').title()} processed sensor data and routed decision."
            ),
            model=model,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            status="success",
        )
        runs.append(run)
        all_tools.extend(_make_tool_calls(run_id, agent, i))

    session.add_all(runs)
    await session.flush()
    session.add_all(all_tools)
    await session.flush()
    print(f"   OK {len(runs)} agent_runs, {len(all_tools)} tool_calls")

    # ── RAG queries ───────────────────────────────────────────────────────────
    rag_queries = [
        RagQuery(
            agent_run_id=RUN_BASE_ID.format(1),
            query="bearing wear maintenance procedure",
            top_k=5, retrieval_latency_ms=145, avg_similarity=0.82,
            sources=[{"doc": "maintenance_manual.pdf", "page": 34},
                     {"doc": "iso13374.pdf", "page": 12}],
            hit=True,
        ),
        RagQuery(
            agent_run_id=RUN_BASE_ID.format(2),
            query="centrifugal pump vibration threshold standards",
            top_k=5, retrieval_latency_ms=198, avg_similarity=0.76,
            sources=[{"doc": "iso10816.pdf", "page": 7}],
            hit=True,
        ),
        RagQuery(
            agent_run_id=RUN_BASE_ID.format(7),
            query="economic impact downtime calculation",
            top_k=5, retrieval_latency_ms=112, avg_similarity=0.91,
            sources=[{"doc": "cost_model.md", "page": 1}],
            hit=True,
        ),
        RagQuery(
            query="induction motor overheating causes",
            top_k=5, retrieval_latency_ms=203, avg_similarity=0.61,
            sources=[], hit=False,
        ),
        RagQuery(
            query="predictive maintenance schedule optimization",
            top_k=5, retrieval_latency_ms=167, avg_similarity=0.79,
            sources=[{"doc": "pm_guide.pdf", "page": 22}],
            hit=True,
        ),
    ]
    session.add_all(rag_queries)
    await session.flush()
    print(f"   OK {len(rag_queries)} rag_queries (1 miss for metrics)")

    # ── Audit log ─────────────────────────────────────────────────────────────
    audit_entries = [
        AuditLog(
            actor_id=USERS["supervisor"], actor_type="user",
            action="acknowledge", entity_type="alert", entity_id=ALERT_C_ID,
            diff={"status": {"from": "new", "to": "acknowledged"}},
            correlation_id="demo-corr-001",
            created_at=_dt(1.5),
        ),
        AuditLog(
            actor_id=RUN_BASE_ID.format(5), actor_type="agent",
            action="create", entity_type="work_order", entity_id=WO_OPEN_ID,
            diff={"created": True, "priority": "critical", "estimated_cost": 3200},
            correlation_id="demo-corr-001",
            created_at=_dt(1.5),
        ),
        AuditLog(
            actor_id=USERS["maintenance_manager"], actor_type="user",
            action="assign", entity_type="work_order", entity_id=WO_PROG_ID,
            diff={"assigned_to": {"from": None, "to": USERS["maintenance_manager"]},
                  "status": {"from": "open", "to": "in_progress"}},
            created_at=_dt(1),
        ),
    ]
    session.add_all(audit_entries)
    await session.flush()
    print(f"   OK {len(audit_entries)} audit_log entries")

    await session.commit()
    print("\nDemo seed complete. Scenario ready for demo.")


def _role_name(role: str) -> str:
    return {
        "operator":            "Carlos Mendoza",
        "supervisor":          "Ana García",
        "maintenance_manager": "Roberto Silva",
        "plant_director":      "Elena Torres",
        "ai_engineer":         "Juan Ramos",
    }[role]


def _make_tool_calls(run_id: str, agent: str, idx: int) -> list[ToolCall]:
    tool_map: dict[str, list] = {
        "supervisor": [
            ("route_decision", {"next": "sensor_analyst"}, {"routed": True}),
        ],
        "sensor_analyst": [
            ("predict_failure", {"machine_id": "PUMP-001"},
             {"probability": 0.98, "alert_level": "red"}),
            ("predict_rca",     {"machine_id": "PUMP-001"},
             {"failure_mode": "bearing_wear", "confidence": 0.87}),
        ],
        "rul_analyst": [
            ("predict_rul", {"machine_id": "PUMP-001"}, {"hours_remaining": 18}),
        ],
        "economic_analyst": [
            ("economic_impact",
             {"machine_id": "PUMP-001", "failure_mode": "bearing_wear"},
             {"downtime_cost": 12400, "repair_cost": 3200}),
        ],
        "work_order_creator": [
            ("create_work_order",
             {"machine_id": "PUMP-001", "priority": "critical"},
             {"work_order_id": WO_OPEN_ID}),
        ],
        "synthesizer": [
            ("synthesize_response", {"context": "all"}, {"response_length": 450}),
        ],
        "doc_expert": [
            ("search_documentation", {"query": "bearing wear"},
             {"sources": 3, "avg_similarity": 0.82}),
        ],
    }
    calls = tool_map.get(agent, [("generic_tool", {}, {})])
    result = []
    for i, (tool_name, inp, out) in enumerate(calls):
        result.append(ToolCall(
            agent_run_id=run_id,
            tool_name=tool_name,
            started_at=NOW - timedelta(hours=2 - idx * 0.05, milliseconds=i * 200),
            latency_ms=100 + i * 50,
            input=inp,
            output=out,
            confidence=0.87 + i * 0.05,
            status="success",
        ))
    return result


async def main() -> None:
    reset = "--reset" in sys.argv
    async with Session() as session:
        if reset:
            await _reset(session)
        existing = await session.execute(select(Plant).where(Plant.id == PLANT_ID))
        if existing.scalar_one_or_none() and not reset:
            print("Demo data already exists. Use --reset to re-seed.")
            return
        await _seed(session)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
