"""Router SSE — GET /api/v2/events/stream."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.events.publisher import event_stream

router = APIRouter(prefix="/api/v2/events", tags=["events"])


@router.get("/stream", summary="Server-Sent Events stream de eventos en vivo")
async def stream_events():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
