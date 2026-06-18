
"""retriever.py 

— Two-stage retrieval: vector search + cross-encoder reranking.
patron retrieve-then-rerank es un enfoque común en sistemas RAG para mejorar la
precisión de la recuperación. Primero se hace una búsqueda vectorial para obtener 
candidatos relevantes, y luego se usa un modelo cross-encoder para reordenar esos
candidatos basándose en su relevancia con la consulta.


Query → embedding → top-K vecinos → reranking con cross-encoder.
"""
from qdrant_client import AsyncQdrantClient
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
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
        self.embedder = SentenceTransformer(config.embedding_model)
        self.reranker = CrossEncoder(config.reranker_model)
        self.cross_encoder = CrossEncoder(config.reranker_model)
        self.qdrant = AsyncQdrantClient(host=config.qdrant_host,port=config.qdrant_port)

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
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions=[
                FieldCondition(
                    key=f"metadata.{k}",match=MatchValue(value=v)
                )
                  for k, v in filters.items()

            ]

            qdrant_filter = Filter(
                must=conditions
            )
        response = await self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_vec,
            limit=retrieve_k,
            query_filter=qdrant_filter,
        )

        candidates = [{

            "text": r.payload["text"],
            "metadata": r.payload["metadata"],
            "vector_score": r.score,

            }

        for r in response.points

        ]

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)


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
    
