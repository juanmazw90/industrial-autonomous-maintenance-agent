"""
rag_config.py — Configuracion y setup de componentes del sistema RAG .

Documento → Chunking → Embeddings → Qdrant (vector DB)
                                          ↓
Pregunta → Embedding → Búsqueda (top 20) → Reranker (top 3) → LLM → RAGResponse

"""

from dataclasses import dataclass, field
from enum import Enum


# Estrategias de chunking

class ChunkStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"



@dataclass
class RAGConfig():
    # Configuración general
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"

    # Configuración de chunking

    chunk_strategy:ChunkStrategy = ChunkStrategy.RECURSIVE 
    chunk_size: int = 500  # caracteres
    overlap_size: int = 50  # caracteres

    # Retriever

    retrieval_top_k: int = 20
    rerank_top_k: int = 3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    # Generation

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    max_tokens: int = 1024

    #cache

    cache_enabled: bool = True
    cache_ttl: int = 3600  # segundos

    # Qdrant

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"



@dataclass
class Document():
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


@dataclass
class Chunk():
    text = str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    embedding: list[float] = field(default_factory=list)



@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0









