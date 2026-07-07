"""
Etapa 1.2 — RAG metrics instrumentation.

InstrumentedRetriever wraps Retriever.retrieve() and persists
rag_queries rows with latency, avg similarity and hit/miss verdict.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.agents.context import current_run_id
from app.infra.db.base import AsyncSessionLocal
from app.infra.db.models import RagQuery
from app.infra.tasks import spawn
from app.services.retrieval import RetrievedChunk, Retriever

_logger = structlog.get_logger(__name__)

# Similarity threshold below which a retrieval is considered a "miss".
_HIT_THRESHOLD = 0.30


async def _persist_rag_query(
    agent_run_id: str | None,
    query: str,
    top_k: int,
    retrieval_latency_ms: int,
    avg_similarity: float,
    sources: list[str],
    hit: bool,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            db.add(RagQuery(
                agent_run_id=agent_run_id,
                query=query[:500],
                top_k=top_k,
                retrieval_latency_ms=retrieval_latency_ms,
                avg_similarity=round(avg_similarity, 4),
                sources=sources,
                hit=hit,
            ))
            await db.commit()
    except Exception:
        _logger.exception("rag.metrics: persist_rag_query failed")


class InstrumentedRetriever:
    """
    Proxy for Retriever that persists rag_queries metrics after each call.
    Forwards attribute access so it can be used as a drop-in replacement.
    """

    def __init__(self, retriever: Retriever, hit_threshold: float = _HIT_THRESHOLD) -> None:
        self._r = retriever
        self._hit_threshold = hit_threshold

    # Propiedades (no copias en __init__): preservan la carga perezosa del Retriever.
    @property
    def config(self):  # noqa: ANN201
        return self._r.config

    @property
    def embedder(self):  # noqa: ANN201
        return self._r.embedder

    @property
    def qdrant(self):  # noqa: ANN201
        return self._r.qdrant

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        started_at = datetime.now(UTC)
        chunks = await self._r.retrieve(query, top_k=top_k, filters=filters)
        latency_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)

        avg_sim = (
            sum(c.rerank_score for c in chunks) / len(chunks) if chunks else 0.0
        )
        sources = list({c.metadata.get("source", "unknown") for c in chunks})
        hit = avg_sim >= self._hit_threshold and bool(chunks)

        run_id = current_run_id.get()

        # Fire-and-forget — never let metrics write block the retrieval response.
        spawn(
            _persist_rag_query(
                agent_run_id=run_id,
                query=query,
                top_k=len(chunks),
                retrieval_latency_ms=latency_ms,
                avg_similarity=avg_sim,
                sources=sources,
                hit=hit,
            )
        )

        return chunks
