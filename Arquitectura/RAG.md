

# Arquitectura RAG de Produccion

Query → Embedding → Vector Search (Qdrant) → Reranking → LLM Generation → Response with Citations

↑

Redis Cache


## Componentes clave

1 Ingesta: Documentos → chunks → embeddings → Qdrant.

2 Retrieval: Query → embedding → top-K vecinos → reranking con cross-encoder.

3 Generación: Contexto + query → LLM → respuesta con citas \[1\], \[2\].

4 Caché: Redis guarda respuestas por hash de query. Hit rate típico: 30-50%.
