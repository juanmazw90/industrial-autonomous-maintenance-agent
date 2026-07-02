"""
Shared context variables for agent instrumentation.
Propagated automatically by asyncio.to_thread into thread-pool nodes.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TypedDict


class UsageInfo(TypedDict, total=False):
    model: str
    input_tokens: int
    output_tokens: int


# Set by any node that calls an LLM so instrument_node can record tokens + cost.
node_usage: ContextVar[UsageInfo | None] = ContextVar("node_usage", default=None)

# Unique ID of the currently-executing node's AgentRun row.
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# Run ID of the supervisor — becomes parent_run_id for every child agent.
supervisor_run_id: ContextVar[str | None] = ContextVar("supervisor_run_id", default=None)

# Running event loop stored at startup so tool wrappers can fire async DB writes
# from thread-pool workers via asyncio.run_coroutine_threadsafe.
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _event_loop
    _event_loop = loop


def get_event_loop() -> asyncio.AbstractEventLoop | None:
    return _event_loop
