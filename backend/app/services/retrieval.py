
"""retriever.py

— Two-stage retrieval: vector search + cross-encoder reranking.
patron retrieve-then-rerank es un enfoque común en sistemas RAG para mejorar la
precisión de la recuperación. Primero se hace una búsqueda vectorial para obtener
candidatos relevantes, y luego se usa un modelo cross-encoder para reordenar esos
candidatos basándose en su relevancia con la consulta.


Query → embedding → top-K vecinos → reranking con cross-encoder.
"""
from dataclasses import dataclass
from functools import cached_property

from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..services.rag_config import RAGConfig


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    vector_score: float
    rerank_score: float


class Retriever:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.qdrant = AsyncQdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    # Modelos pesados con carga perezosa: importar la app no descarga/carga nada
    @cached_property
    def embedder(self) -> SentenceTransformer:
        return SentenceTransformer(self.config.embedding_model)

    @cached_property
    def reranker(self) -> CrossEncoder:
        return CrossEncoder(self.config.reranker_model)

    async def retrieve(
            self,
            query: str,
            top_k: int | None = None,
            filters: dict | None = None) -> list[RetrievedChunk]:

        top_k = top_k or self.config.rerank_top_k
        retrieve_k = self.config.retrieval_top_k

        query_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()


        qdrant_filter = None
        if filters:
            from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue
            conditions: list[Condition] = [
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = await self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_vec,
            limit=retrieve_k,
            query_filter=qdrant_filter,
        )

        candidates = [
            {
                "text": (r.payload or {}).get("text", ""),
                "metadata": (r.payload or {}).get("metadata", {}),
                "vector_score": r.score,
            }
            for r in response.points
        ]

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)  # type: ignore[arg-type]


        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return [

            RetrievedChunk(
                text=c["text"],
                metadata=c["metadata"],
                vector_score=c["vector_score"],
                rerank_score=c["rerank_score"],
            )
            for c in candidates[:top_k]
        ]

