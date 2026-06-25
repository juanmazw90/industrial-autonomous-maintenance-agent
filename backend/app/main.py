import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .graph import get_graph
from .models import FailurePredictionResponse, IncomingSensorReading, InputQuery
from .services.conversation import ConversationStore
from .services.ingestion import IngestionPipeline, parse_document
from .services.predictor import FailurePredictor
from .services.rca_predictor import RCAPredictor
from .services.rag_config import RAGConfig
from .services.retrieval import Retriever
from .services.semantic_cache import SemanticCache

REPO_ROOT  = Path(__file__).resolve().parents[2]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
DATA_PATH  = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))

config    = RAGConfig()
retriever = Retriever(config)                          # carga SentenceTransformer + Qdrant client
pipeline  = IngestionPipeline(config)
predictor     = FailurePredictor()
rca_predictor = RCAPredictor()
graph         = get_graph(config, predictor, retriever, rca_predictor)
store     = ConversationStore()
sem_cache = SemanticCache(retriever.embedder, retriever.qdrant)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar semantic cache (crea colección en Qdrant si no existe)
    try:
        await sem_cache.initialize()
    except Exception as e:
        print(f"[SemanticCache] No se pudo inicializar: {e}")

    # Inicializar predictor en thread pool (CPU-bound, ~30s)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, predictor.initialize, MLFLOW_URI, DATA_PATH)
    except Exception as e:
        print(f"[Predictor] No se pudo inicializar: {e}. /predict/failure no estará disponible.")
    try:
        await loop.run_in_executor(None, rca_predictor.initialize, MLFLOW_URI, DATA_PATH)
    except Exception as e:
        print(f"[RCAPredictor] No se pudo inicializar: {e}. Diagnóstico RCA no estará disponible.")
    yield


app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    cache_stats = await sem_cache.stats()
    return {
        "status":              "ok",
        "version":             "0.3.0",
        "predictor_ready":     predictor.initialized,
        "rca_predictor_ready": rca_predictor.initialized,
        "semantic_cache":      cache_stats,
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Recibe un archivo, lo parsea, genera embeddings y almacena en Qdrant."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    try:
        doc = parse_document(content, file.filename or "documento")
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    loop = asyncio.get_event_loop()
    try:
        n_chunks = await loop.run_in_executor(None, pipeline.ingest, doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la ingestión: {e}")

    return {
        "doc_id":        doc.doc_id,
        "source":        file.filename,
        "pages":         doc.metadata.get("pages", 1),
        "chunks_stored": n_chunks,
    }


@app.post("/process_input")
async def process_input(user_input: InputQuery) -> dict:
    history = store.get_history(user_input.session_id)

    # ── Semantic cache check ──────────────────────────────────────────────
    cached = await sem_cache.get(user_input.query)
    if cached:
        # Hit: devolver respuesta sin llamar a LangGraph
        store.append_turn(user_input.session_id, user_input.query, cached["response"])
        return {
            "response":   cached["response"],
            "sources":    cached["sources"],
            "agent_used": cached["agent_used"],
            "session_id": user_input.session_id,
            "cached":     True,
            "similarity": cached.get("similarity"),
        }

    # ── LangGraph invocation ──────────────────────────────────────────────
    try:
        result = await graph.ainvoke({
            "query":                user_input.query,
            "conversation_history": history,
            "retrieved_docs":       [],
            "sensor_analysis":      None,
            "economic_impact":      None,
            "work_order":           None,
            "next_agent":           "",
            "final_response":       "",
            "sources":              [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    agent_used = result.get("next_agent", "unknown")
    response   = result["final_response"]
    sources    = result["sources"]

    store.append_turn(user_input.session_id, user_input.query, response)

    # ── Store in semantic cache (solo doc_expert) ─────────────────────────
    await sem_cache.set(user_input.query, response, sources, agent_used)

    return {
        "response":   response,
        "sources":    sources,
        "agent_used": agent_used,
        "session_id": user_input.session_id,
        "cached":     False,
    }


@app.post("/predict/failure", response_model=FailurePredictionResponse)
async def predict_failure(machine_id: str) -> dict:
    """Predice si una máquina fallará en las próximas 24 horas."""
    if not predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor no está disponible.")
    try:
        return predictor.predict(machine_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predict/failure/all")
async def predict_all_machines() -> list[dict]:
    """Devuelve predicciones para todas las máquinas conocidas."""
    if not predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor no está disponible.")
    return predictor.predict_all()


@app.post("/sensors/reading", response_model=FailurePredictionResponse)
async def receive_sensor_reading(reading: IncomingSensorReading) -> dict:
    """Ingesta una lectura de sensores en tiempo real y devuelve la predicción actualizada."""
    if not predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor no está disponible.")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, predictor.update_with_reading, reading.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
