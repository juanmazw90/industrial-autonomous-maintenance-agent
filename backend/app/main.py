import asyncio

from dotenv import load_dotenv
load_dotenv()  # carga .env antes de que los agentes lean ANTHROPIC_API_KEY

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .graph import get_graph
from .models import InputQuery
from .services.conversation import ConversationStore
from .services.ingestion import IngestionPipeline, parse_document
from .services.rag_config import RAGConfig

config = RAGConfig()
pipeline = IngestionPipeline(config)   # instancia única — carga el modelo de embeddings una vez
graph = get_graph(config)
store = ConversationStore()

app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """
    Recibe un archivo (PDF, Markdown, TXT), lo parsea, genera embeddings
    y almacena los chunks en Qdrant.

    El pipeline es síncrono (SentenceTransformer + QdrantClient sync),
    así que lo ejecutamos en un thread pool para no bloquear el event loop.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        doc = parse_document(content, file.filename or "documento")
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    # run_in_executor → ejecuta código síncrono sin bloquear el event loop async
    loop = asyncio.get_event_loop()
    try:
        n_chunks = await loop.run_in_executor(None, pipeline.ingest, doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la ingestión: {e}")

    return {
        "doc_id": doc.doc_id,
        "source": file.filename,
        "pages": doc.metadata.get("pages", 1),
        "chunks_stored": n_chunks,
    }


@app.post("/process_input")
async def process_input(user_input: InputQuery) -> dict:
    history = store.get_history(user_input.session_id)

    try:
        result = await graph.ainvoke({
            "query": user_input.query,
            "conversation_history": history,
            "retrieved_docs": [],
            "sensor_analysis": None,
            "economic_impact": None,
            "work_order": None,
            "next_agent": "",
            "final_response": "",
            "sources": [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    store.append_turn(user_input.session_id, user_input.query, result["final_response"])

    return {
        "response": result["final_response"],
        "sources": result["sources"],
        "agent_used": result.get("next_agent", "unknown"),
        "session_id": user_input.session_id,
    }


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
