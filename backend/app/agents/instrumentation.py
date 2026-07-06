"""
Etapa 1.1 — Agent instrumentation.

instrument_node  — async decorator for LangGraph nodes (sync and async).
instrument_tool  — sync decorator for tool functions called inside nodes.

Both persist rows in PostgreSQL (agent_runs, tool_calls) and publish
agent_run.finished events to Redis without ever breaking the agent logic.
All errors are swallowed so instrumentation never crashes the graph.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import structlog

from app.agents.context import (
    current_run_id,
    get_event_loop,
    node_usage,
    supervisor_run_id,
)
from app.events.publisher import publish
from app.events.schemas import AgentRunFinishedEvent
from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import AgentRun, ToolCall

_logger = structlog.get_logger(__name__)


# ── Pricing table (USD per million tokens, [input, output]) ──────────────────

_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
}


def _estimate_cost(model: str | None, in_tok: int, out_tok: int) -> float | None:
    if not model or model not in _PRICE_PER_M:
        return None
    ip, op = _PRICE_PER_M[model]
    return round((in_tok * ip + out_tok * op) / 1_000_000, 6)


# ── Async DB helpers ─────────────────────────────────────────────────────────

async def _create_agent_run(
    run_id: str,
    agent_name: str,
    session_id: str,
    parent_run_id: str | None,
    input_summary: str | None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            db.add(AgentRun(
                id=run_id,
                session_id=session_id,
                agent_name=agent_name,
                parent_run_id=parent_run_id,
                started_at=datetime.now(timezone.utc),
                input_summary=input_summary,
                status="running",
            ))
            await db.commit()
    except Exception:
        _logger.exception("instrumentation: create_agent_run failed", run_id=run_id)


async def _finish_agent_run(
    run_id: str,
    status: str,
    latency_ms: int,
    output_summary: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    error: str | None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.finished_at = datetime.now(timezone.utc)
                run.latency_ms = latency_ms
                run.status = status
                run.output_summary = output_summary
                run.model = model
                run.input_tokens = input_tokens
                run.output_tokens = output_tokens
                run.cost_usd = cost_usd
                run.error = error
                await db.commit()
    except Exception:
        _logger.exception("instrumentation: finish_agent_run failed", run_id=run_id)


async def _create_tool_call(
    agent_run_id: str,
    tool_name: str,
    input_data: Any,
    output_data: Any,
    latency_ms: int,
    status: str,
    error: str | None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            db.add(ToolCall(
                id=str(uuid.uuid4()),
                agent_run_id=agent_run_id,
                tool_name=tool_name,
                started_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                input={"arg": str(input_data)[:400]} if input_data is not None else None,
                output={"result": str(output_data)[:400]} if output_data is not None else None,
                status=status,
                error=error,
            ))
            await db.commit()
    except Exception:
        _logger.exception("instrumentation: create_tool_call failed", tool_name=tool_name)


def _fire_tool_call(
    agent_run_id: str,
    tool_name: str,
    input_data: Any,
    output_data: Any,
    latency_ms: int,
    status: str,
    error: str | None,
) -> None:
    """Schedule tool call DB write onto the main event loop from a worker thread."""
    loop = get_event_loop()
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            _create_tool_call(agent_run_id, tool_name, input_data, output_data, latency_ms, status, error),
            loop,
        )


# ── Node instrumentation ─────────────────────────────────────────────────────

def instrument_node(name: str) -> Callable:
    """
    Decorator factory.  Works for both sync and async LangGraph nodes.

    Wraps every node as async so LangGraph's async executor handles it uniformly.
    Sync node bodies run via asyncio.to_thread(), which copies the ContextVar
    snapshot to the thread — so instrument_tool can read current_run_id.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(state: dict) -> dict:
            run_id = str(uuid.uuid4())
            session_id = state.get("session_id", "no-session")
            parent = supervisor_run_id.get() if name != "supervisor" else None
            input_summary = str(state.get("query", ""))[:300] or None

            await _create_agent_run(run_id, name, session_id, parent, input_summary)

            tok_run = current_run_id.set(run_id)
            tok_sup = supervisor_run_id.set(run_id) if name == "supervisor" else None
            # Reset node_usage so the node can set a fresh value
            tok_usg = node_usage.set(None)

            started_at = datetime.now(timezone.utc)
            status = "success"
            error_msg: str | None = None
            result: dict = {}

            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(state)
                else:
                    result = await asyncio.to_thread(fn, state)
                return result

            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:500]
                raise

            finally:
                finished_at = datetime.now(timezone.utc)
                latency_ms = int((finished_at - started_at).total_seconds() * 1000)

                usage = node_usage.get()
                model = usage.get("model") if usage else None
                in_tok = usage.get("input_tokens", 0) if usage else 0
                out_tok = usage.get("output_tokens", 0) if usage else 0
                cost = _estimate_cost(model, in_tok, out_tok)

                output_summary = None
                if isinstance(result, dict):
                    for key in ("final_response", "sensor_analysis", "rul_prediction", "next_agent"):
                        val = result.get(key)
                        if val:
                            output_summary = str(val)[:300]
                            break

                await _finish_agent_run(
                    run_id=run_id,
                    status=status,
                    latency_ms=latency_ms,
                    output_summary=output_summary,
                    model=model,
                    input_tokens=in_tok or None,
                    output_tokens=out_tok or None,
                    cost_usd=cost,
                    error=error_msg,
                )

                try:
                    await publish(AgentRunFinishedEvent(
                        correlation_id=run_id,
                        payload={
                            "agent": name,
                            "latency_ms": latency_ms,
                            "status": status,
                            "session_id": session_id,
                        },
                    ))
                except Exception:
                    pass

                current_run_id.reset(tok_run)
                node_usage.reset(tok_usg)
                if tok_sup is not None:
                    supervisor_run_id.reset(tok_sup)

        return wrapper
    return decorator


# ── Tool instrumentation ─────────────────────────────────────────────────────

def instrument_tool(name: str) -> Callable:
    """
    Decorator factory for sync tool functions (predict_fn, search_fn, etc.).

    Captures latency and outcome; fires an async DB write back to the main
    event loop via asyncio.run_coroutine_threadsafe without blocking the tool.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = datetime.now(timezone.utc)
            run_id = current_run_id.get()
            output_data: Any = None
            status = "success"
            error_msg: str | None = None

            try:
                output_data = fn(*args, **kwargs)
                return output_data
            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:300]
                raise
            finally:
                latency_ms = int(
                    (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                )
                if run_id:
                    _fire_tool_call(
                        agent_run_id=run_id,
                        tool_name=name,
                        input_data=args[0] if args else kwargs,
                        output_data=output_data,
                        latency_ms=latency_ms,
                        status=status,
                        error=error_msg,
                    )

        return wrapper
    return decorator


# ── Async tool instrumentation ───────────────────────────────────────────────

def instrument_async_tool(name: str) -> Callable:
    """Same as instrument_tool but for async tool functions (e.g. search_documentation)."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = datetime.now(timezone.utc)
            run_id = current_run_id.get()
            output_data: Any = None
            status = "success"
            error_msg: str | None = None

            try:
                output_data = await fn(*args, **kwargs)
                return output_data
            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:300]
                raise
            finally:
                latency_ms = int(
                    (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                )
                if run_id:
                    await _create_tool_call(
                        agent_run_id=run_id,
                        tool_name=name,
                        input_data=args[0] if args else kwargs,
                        output_data=output_data,
                        latency_ms=latency_ms,
                        status=status,
                        error=error_msg,
                    )

        return wrapper
    return decorator
