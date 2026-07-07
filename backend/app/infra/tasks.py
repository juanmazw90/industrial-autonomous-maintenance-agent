"""Fire-and-forget seguro para corutinas de fondo.

asyncio.create_task() sin retener la referencia permite que el GC cancele
el task a mitad de ejecución. spawn() retiene la referencia hasta que termina.
"""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_background_tasks: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
