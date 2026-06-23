"""
semantic_cache.py — Caché semántica para respuestas del agente RAG.

Diferencia con el exact cache (cache.py):
  - Exact cache: "¿qué es el desgaste?" vs "que es el desgaste" → mismo hash → hit
  - Semantic cache: "¿qué es el desgaste?" vs "explícame el desgaste" → similitud
    coseno > 0.95 → misma respuesta cacheada sin llamar al LLM

Implementación:
  - Embeddings: SentenceTransformer all-MiniLM-L6-v2 (384 dims)
    compartido con el Retriever — no se carga el modelo dos veces
  - Almacenamiento: Qdrant colección "query_cache" (vector + payload)
  - TTL: timestamp en payload, verificado en lectura (Qdrant no tiene TTL nativo)
  - Solo cachea respuestas de doc_expert — las predicciones cambian en tiempo real
    y nunca deben servirse desde caché

Flujo:
  get(query) → embed → search Qdrant (cosine ≥ 0.95) → check TTL → return | None
  set(query, response) → embed → upsert Qdrant con payload
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

COLLECTION       = "query_cache"
SIMILARITY_THRESHOLD = 0.95
TTL_SECONDS      = 3600   # 1 hora
VECTOR_DIM       = 384


class SemanticCache:
    def __init__(
        self,
        embedder: SentenceTransformer,
        qdrant: AsyncQdrantClient,
        ttl: int = TTL_SECONDS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedder  = embedder
        self._qdrant    = qdrant
        self._ttl       = ttl
        self._threshold = threshold
        self._hits      = 0
        self._misses    = 0

    # ── Setup ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Crea la colección query_cache en Qdrant si no existe."""
        existing = [c.name for c in (await self._qdrant.get_collections()).collections]
        if COLLECTION not in existing:
            await self._qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"[SemanticCache] Colección '{COLLECTION}' creada en Qdrant.")
        else:
            print(f"[SemanticCache] Colección '{COLLECTION}' ya existe.")

    # ── Operaciones principales ────────────────────────────────────────────

    async def get(self, query: str) -> dict[str, Any] | None:
        """
        Busca una respuesta semánticamente similar para la query.
        Devuelve el dict de respuesta cacheado o None si es un miss.
        """
        vector = self._embed(query)

        results: list[ScoredPoint] = (
            await self._qdrant.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=1,
                with_payload=True,
            )
        ).points

        if not results:
            self._misses += 1
            return None

        hit = results[0]

        # Verificar umbral de similitud
        if hit.score < self._threshold:
            self._misses += 1
            return None

        # Verificar TTL
        cached_at = hit.payload.get("cached_at", 0)
        if time.time() - cached_at > self._ttl:
            self._misses += 1
            return None

        self._hits += 1
        return {
            "response":   hit.payload["response"],
            "sources":    hit.payload.get("sources", []),
            "agent_used": hit.payload.get("agent_used", "doc_expert"),
            "cached":     True,
            "similarity": round(hit.score, 4),
        }

    async def set(self, query: str, response: str, sources: list, agent_used: str) -> None:
        """
        Almacena la respuesta en la caché semántica.
        Solo persiste respuestas de doc_expert — no cachea predicciones de sensores.
        """
        if agent_used != "doc_expert":
            return

        vector    = self._embed(query)
        point_id  = _query_to_uuid(query)

        await self._qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "query":      query,
                        "response":   response,
                        "sources":    sources,
                        "agent_used": agent_used,
                        "cached_at":  time.time(),
                    },
                )
            ],
        )

    # ── Métricas ───────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total > 0 else 0.0

    async def stats(self) -> dict:
        try:
            count = (await self._qdrant.get_collection(COLLECTION)).points_count
        except Exception:
            count = 0
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   self.hit_rate,
            "cached_queries": count,
            "ttl_seconds":    self._ttl,
            "threshold":      self._threshold,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text, normalize_embeddings=True).tolist()


def _query_to_uuid(query: str) -> str:
    """Genera un UUID v4 determinista a partir del hash de la query."""
    digest = hashlib.md5(query.strip().lower().encode()).hexdigest()
    # Formatear como UUID para que Qdrant lo acepte como string ID
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
