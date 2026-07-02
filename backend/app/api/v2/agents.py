"""
Etapa 2.2 — Agents / AI Control Center routers.

GET /api/v2/agents/summary           — p50/p95 latency, calls, success rate, cost per agent
GET /api/v2/agents/runs              — paginated list with filters
GET /api/v2/agents/runs/{run_id}     — full run detail with tool calls
GET /api/v2/agents/runs/{run_id}/trace — supervisor→agents→tools tree
GET /api/v2/agents/evaluations       — list evaluations
POST /api/v2/agents/evaluations      — create evaluation for a run
"""
from __future__ import annotations

import statistics
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import AgentEvaluation, AgentRun, RagQuery, ToolCall

router = APIRouter(prefix="/api/v2/agents", tags=["agents"])


# ── /agents/summary ────────────────────────────────────────────────────────────

@router.get("/summary")
async def agents_summary() -> dict[str, Any]:
    """Aggregated stats per agent: latency percentiles, call counts, cost, tokens."""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(
                AgentRun.agent_name,
                AgentRun.status,
                AgentRun.latency_ms,
                AgentRun.cost_usd,
                AgentRun.input_tokens,
                AgentRun.output_tokens,
            )
        )).all()

    # Group by agent_name
    by_agent: dict[str, list] = {}
    for row in rows:
        by_agent.setdefault(row.agent_name, []).append(row)

    summary = []
    for agent_name, runs in by_agent.items():
        latencies = sorted(r.latency_ms for r in runs if r.latency_ms is not None)
        success = sum(1 for r in runs if r.status == "success")
        total = len(runs)
        cost_total = sum(r.cost_usd or 0.0 for r in runs)
        in_tokens = sum(r.input_tokens or 0 for r in runs)
        out_tokens = sum(r.output_tokens or 0 for r in runs)

        def _percentile(lst: list[int], pct: float) -> int | None:
            if not lst:
                return None
            idx = int(len(lst) * pct / 100)
            return lst[min(idx, len(lst) - 1)]

        summary.append({
            "agent_name": agent_name,
            "total_calls": total,
            "success_rate": round(success / total, 3) if total else 0.0,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "cost_usd_total": round(cost_total, 6),
            "input_tokens_total": in_tokens,
            "output_tokens_total": out_tokens,
        })

    # Overall totals row
    all_runs = list(rows)
    all_latencies = sorted(r.latency_ms for r in all_runs if r.latency_ms is not None)

    def _p(lst, pct):
        if not lst:
            return None
        idx = int(len(lst) * pct / 100)
        return lst[min(idx, len(lst) - 1)]

    return {
        "by_agent": sorted(summary, key=lambda x: x["agent_name"]),
        "totals": {
            "total_calls": len(all_runs),
            "success_rate": round(
                sum(1 for r in all_runs if r.status == "success") / len(all_runs), 3
            ) if all_runs else 0.0,
            "cost_usd_total": round(sum(r.cost_usd or 0.0 for r in all_runs), 6),
            "latency_p50_ms": _p(all_latencies, 50),
            "latency_p95_ms": _p(all_latencies, 95),
        },
    }


# ── /agents/runs ───────────────────────────────────────────────────────────────

@router.get("/runs")
async def list_runs(
    agent: str | None = Query(None),
    status: str | None = Query(None),
    session_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Paginated list of agent runs with optional filters."""
    async with AsyncSessionLocal() as db:
        q = select(AgentRun).order_by(AgentRun.started_at.desc())
        if agent:
            q = q.where(AgentRun.agent_name == agent)
        if status:
            q = q.where(AgentRun.status == status)
        if session_id:
            q = q.where(AgentRun.session_id == session_id)

        count_q = select(func.count(AgentRun.id))
        if agent:
            count_q = count_q.where(AgentRun.agent_name == agent)
        if status:
            count_q = count_q.where(AgentRun.status == status)
        if session_id:
            count_q = count_q.where(AgentRun.session_id == session_id)

        total = (await db.execute(count_q)).scalar() or 0
        runs = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "agent_name": r.agent_name,
                "parent_run_id": r.parent_run_id,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "model": r.model,
                "started_at": r.started_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "error": r.error,
            }
            for r in runs
        ],
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Full detail for a single agent run including tool calls and RAG queries."""
    async with AsyncSessionLocal() as db:
        run = (await db.execute(
            select(AgentRun).where(AgentRun.id == run_id)
        )).scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        tool_calls = (await db.execute(
            select(ToolCall)
            .where(ToolCall.agent_run_id == run_id)
            .order_by(ToolCall.started_at)
        )).scalars().all()

        rag_queries = (await db.execute(
            select(RagQuery)
            .where(RagQuery.agent_run_id == run_id)
            .order_by(RagQuery.created_at)
        )).scalars().all()

    return {
        "id": run.id,
        "session_id": run.session_id,
        "agent_name": run.agent_name,
        "parent_run_id": run.parent_run_id,
        "status": run.status,
        "input_summary": run.input_summary,
        "output_summary": run.output_summary,
        "reasoning_summary": run.reasoning_summary,
        "model": run.model,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": run.cost_usd,
        "latency_ms": run.latency_ms,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
        "retries": run.retries,
        "tool_calls": [
            {
                "id": tc.id,
                "tool_name": tc.tool_name,
                "status": tc.status,
                "latency_ms": tc.latency_ms,
                "input": tc.input,
                "output": tc.output,
                "error": tc.error,
                "started_at": tc.started_at.isoformat(),
            }
            for tc in tool_calls
        ],
        "rag_queries": [
            {
                "id": rq.id,
                "query": rq.query,
                "top_k": rq.top_k,
                "avg_similarity": rq.avg_similarity,
                "retrieval_latency_ms": rq.retrieval_latency_ms,
                "hit": rq.hit,
                "sources": rq.sources,
                "created_at": rq.created_at.isoformat(),
            }
            for rq in rag_queries
        ],
    }


@router.get("/runs/{run_id}/trace")
async def get_run_trace(run_id: str) -> dict[str, Any]:
    """
    Supervisor→agents→tools execution tree for a session.
    Finds the supervisor run (or the given run), then fetches all child runs
    and their tool calls to build the trace tree.
    """
    async with AsyncSessionLocal() as db:
        # Start from the given run; walk up to find the root supervisor
        root = (await db.execute(
            select(AgentRun).where(AgentRun.id == run_id)
        )).scalar_one_or_none()
        if not root:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        # Walk to root (supervisor has no parent_run_id)
        current = root
        while current.parent_run_id:
            parent = (await db.execute(
                select(AgentRun).where(AgentRun.id == current.parent_run_id)
            )).scalar_one_or_none()
            if not parent:
                break
            current = parent
        supervisor_run = current

        # All runs in the same session with this supervisor as root (or children of it)
        all_runs_in_session = (await db.execute(
            select(AgentRun)
            .where(AgentRun.session_id == supervisor_run.session_id)
            .order_by(AgentRun.started_at)
        )).scalars().all()

        # Tool calls per run
        run_ids = [r.id for r in all_runs_in_session]
        all_tool_calls = (await db.execute(
            select(ToolCall)
            .where(ToolCall.agent_run_id.in_(run_ids))
            .order_by(ToolCall.started_at)
        )).scalars().all()

    tools_by_run: dict[str, list] = {}
    for tc in all_tool_calls:
        tools_by_run.setdefault(tc.agent_run_id, []).append({
            "id": tc.id,
            "tool_name": tc.tool_name,
            "status": tc.status,
            "latency_ms": tc.latency_ms,
            "started_at": tc.started_at.isoformat(),
        })

    def _build_node(run: AgentRun) -> dict:
        return {
            "id": run.id,
            "agent_name": run.agent_name,
            "status": run.status,
            "latency_ms": run.latency_ms,
            "cost_usd": run.cost_usd,
            "started_at": run.started_at.isoformat(),
            "tool_calls": tools_by_run.get(run.id, []),
            "children": [
                _build_node(child)
                for child in all_runs_in_session
                if child.parent_run_id == run.id
            ],
        }

    return {
        "session_id": supervisor_run.session_id,
        "trace": _build_node(supervisor_run),
    }


# ── /agents/evaluations ────────────────────────────────────────────────────────

class EvaluationCreate(BaseModel):
    agent_run_id: str
    accuracy: float | None = None
    groundedness: float | None = None
    hallucination_flag: bool = False
    tool_success_rate: float | None = None
    evaluator: str | None = None
    notes: str | None = None


@router.get("/evaluations")
async def list_evaluations(
    agent_run_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        q = select(AgentEvaluation).order_by(AgentEvaluation.created_at.desc())
        if agent_run_id:
            q = q.where(AgentEvaluation.agent_run_id == agent_run_id)

        total = (await db.execute(
            select(func.count(AgentEvaluation.id))
            .where(AgentEvaluation.agent_run_id == agent_run_id)
            if agent_run_id
            else select(func.count(AgentEvaluation.id))
        )).scalar() or 0

        evals = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "evaluations": [
            {
                "id": e.id,
                "agent_run_id": e.agent_run_id,
                "accuracy": e.accuracy,
                "groundedness": e.groundedness,
                "hallucination_flag": e.hallucination_flag,
                "tool_success_rate": e.tool_success_rate,
                "evaluator": e.evaluator,
                "notes": e.notes,
                "created_at": e.created_at.isoformat(),
            }
            for e in evals
        ],
    }


@router.post("/evaluations", status_code=201)
async def create_evaluation(body: EvaluationCreate) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        # Validate run exists
        run = (await db.execute(
            select(AgentRun).where(AgentRun.id == body.agent_run_id)
        )).scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail=f"AgentRun '{body.agent_run_id}' not found")

        ev = AgentEvaluation(
            agent_run_id=body.agent_run_id,
            accuracy=body.accuracy,
            groundedness=body.groundedness,
            hallucination_flag=body.hallucination_flag,
            tool_success_rate=body.tool_success_rate,
            evaluator=body.evaluator,
            notes=body.notes,
        )
        db.add(ev)
        await db.commit()
        await db.refresh(ev)

    return {
        "id": ev.id,
        "agent_run_id": ev.agent_run_id,
        "created_at": ev.created_at.isoformat(),
    }
