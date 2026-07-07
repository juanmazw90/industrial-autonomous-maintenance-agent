"""
rag_config.py — Configuracion y setup de componentes del sistema RAG .

Documento → Chunking → Embeddings → Qdrant (vector DB)
                                          ↓
Pregunta → Embedding → Búsqueda (top 20) → Reranker (top 3) → LLM → RAGResponse

"""

from dataclasses import dataclass, field
from enum import Enum

from app.infra.settings import settings

# Estrategias de chunking

class ChunkStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"



@dataclass
class RAGConfig:
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


    # Generation — Anthropic Claude
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1
    max_tokens: int = 1024

    #cache

    cache_enabled: bool = True
    cache_ttl: int = 3600  # segundos

    # Qdrant — defaults desde settings (única fuente de verdad)

    qdrant_host: str = field(default_factory=lambda: settings.qdrant_host)
    qdrant_port: int = field(default_factory=lambda: settings.qdrant_port)
    collection_name: str = "documents"



@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


@dataclass
class Chunk:
    text: str = ""
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    embedding: list[float] = field(default_factory=list)



@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0









