"""
ingestion.py — Pipeline de ingesta de documentos (PDF, Markdown, texto).

Flujo:  archivo (bytes) → Document → chunks → embeddings → Qdrant

La clase IngestionPipeline es síncrona porque la ingestión ocurre en
segundo plano (no es un path de alta frecuencia). El endpoint FastAPI
la llama con run_in_executor para no bloquear el event loop.
"""

import hashlib
import uuid
from functools import cached_property
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .rag_config import Chunk, Document, RAGConfig

# ---------------------------------------------------------------------------
# Parsers de documentos
# ---------------------------------------------------------------------------

def parse_pdf(content: bytes, filename: str) -> Document:
    """Extrae texto de un PDF usando PyMuPDF y devuelve un Document."""
    doc_fitz = fitz.open(stream=content, filetype="pdf")
    pages_text = []
    for page in doc_fitz:
        text = page.get_text().strip()
        if text:
            pages_text.append(text)
    full_text = "\n\n".join(pages_text)
    doc_id = hashlib.md5(content).hexdigest()
    return Document(
        content=full_text,
        metadata={"source": filename, "type": "pdf", "pages": len(pages_text)},
        doc_id=doc_id,
    )


def parse_markdown(content: bytes, filename: str) -> Document:
    text = content.decode("utf-8")
    doc_id = hashlib.md5(content).hexdigest()
    return Document(
        content=text,
        metadata={"source": filename, "type": "markdown"},
        doc_id=doc_id,
    )


def parse_text(content: bytes, filename: str) -> Document:
    text = content.decode("utf-8", errors="replace")
    doc_id = hashlib.md5(content).hexdigest()
    return Document(
        content=text,
        metadata={"source": filename, "type": "text"},
        doc_id=doc_id,
    )


PARSERS = {
    ".pdf": parse_pdf,
    ".md":  parse_markdown,
    ".txt": parse_text,
}


def parse_document(content: bytes, filename: str) -> Document:
    """Selecciona el parser correcto según la extensión del archivo."""
    ext = Path(filename).suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Formato no soportado: '{ext}'. Formatos válidos: {list(PARSERS)}")
    return parser(content, filename)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class IngestionPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap_size,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._collection_ready = False

    # Carga perezosa: construir el pipeline no descarga modelos ni toca Qdrant
    @cached_property
    def embedder(self) -> SentenceTransformer:
        return SentenceTransformer(self.config.embedding_model)

    @cached_property
    def qdrant(self) -> QdrantClient:
        return QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)

    def _ensure_collection(self) -> None:
        """Crea la colección en Qdrant si no existe (una sola vez)."""
        if self._collection_ready:
            return
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.config.collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        self._collection_ready = True

    def chunk_document(self, doc: Document) -> list[Chunk]:
        texts = self.splitter.split_text(doc.content)
        chunks = []
        for i, text in enumerate(texts):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.doc_id}:{i}"))
            chunks.append(Chunk(
                text=text,
                metadata={
                    **doc.metadata,
                    "doc_id": doc.doc_id,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
                chunk_id=chunk_id,
            ))
        return chunks

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts, batch_size=64, normalize_embeddings=True
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        return chunks

    def store_chunks(self, chunks: list[Chunk]) -> None:
        points = [
            PointStruct(
                id=c.chunk_id,
                vector=c.embedding,
                payload={"text": c.text, "metadata": c.metadata},
            )
            for c in chunks
        ]
        self.qdrant.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

    def ingest(self, doc: Document) -> int:
        """Procesa un Document completo y lo almacena en Qdrant. Devuelve nº de chunks."""
        self._ensure_collection()
        chunks = self.chunk_document(doc)
        chunks = self.embed_chunks(chunks)
        self.store_chunks(chunks)
        return len(chunks)
