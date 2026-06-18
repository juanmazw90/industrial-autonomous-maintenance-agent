"""
Diseño de arquitectura del sistema RAG 
Componentes: (ingesta, retrieval, generación, caché), las interfaces entre ellos, y el flujo de datos end-to-end."
Un RAG de producción Necesita componentes desacoplados para poder escalar, testear, y reemplazar partes sin romper todo.

ingestion.py — Pipeline de Ingestion de Documentos con chunking inteligente.

Ingesta: Documentos → chunks → embeddings → Qdrant.

"""
import hashlib
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ..services.rag_config import Chunk, Document, RAGConfig



class IngestionPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        self.qdrant = QdrantClient(host=config.qdrant_host,port=config.qdrant_port)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size,
        chunk_overlap=config.overlap_size,
        separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._ensure_collection()
    
 
    def _ensure_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.config.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def chunk_document(self, doc: Document) -> List[Chunk]:
        texts = self.splitter.split_text(doc.content)
        chunks = []
        for i, text in enumerate(texts):
            chunk_id = hashlib.sha256(f"{doc.id}_{i}".encode()).hexdigest()
            chunks.append(Chunk(text=text,
                                metadata={
                                **doc.metadata,
                                "doc_id": doc.doc_id,
                                "chunk_index": i,
                                "total_chunks": len(texts),
                            },
                            chunk_id=chunk_id,
                        ))
        return chunks

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
                texts, batch_size=64, normalize_embeddings=True
        
    )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        return chunks
    
    def store_chunks(self, chunks: list[Chunk]):
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
        chunks = self.chunk_document(doc)
        chunks = self.embed_chunks(chunks)
        self.store_chunks(chunks)
        return len(chunks)

    

    
def load_markdown_files(directory: str) -> list[Document]:
    docs = []
    for path in Path(directory).glob("**/*.md"):
        content = path.read_text(encoding="utf-8")
        docs.append(Document(
            content=content,
            metadata={"source": str(path), "type": "markdown"},
            doc_id=hashlib.md5(str(path).encode()).hexdigest(),
        ))
    return docs
    
if __name__ == "__main__":
    config = RAGConfig()
    pipeline = IngestionPipeline(config)
    docs = load_markdown_files(".data/docs")
    n = pipeline.ingest(docs)
    print(f"Ingested {len(docs)} documents into {n} chunks")