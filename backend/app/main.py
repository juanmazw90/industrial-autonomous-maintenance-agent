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
from .services.cmms import CMMS
from .services.rca_predictor import RCAPredictor
from .services.rul_predictor import RULPredictor
from .services.rag_config import RAGConfig
from .services.retrieval import Retriever
from .services.semantic_cache import SemanticCache

# ── Langfuse (opcional — activo cuando LANGFUSE_PUBLIC_KEY está en el entorno) ──
try:
    from langfuse import Langfuse as _Langfuse
    _LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    _langfuse: "_Langfuse | None" = (
        _Langfuse(
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        )
        if _LANGFUSE_ENABLED
        else None
    )
except Exception:
    _LANGFUSE_ENABLED = False
    _langfuse = None


REPO_ROOT  = Path(__file__).resolve().parents[2]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
DATA_PATH  = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))

config    = RAGConfig()
retriever = Retriever(config)
pipeline  = IngestionPipeline(config)
predictor     = FailurePredictor()
rca_predictor = RCAPredictor()
rul_predictor = RULPredictor()
cmms          = CMMS()
graph         = get_graph(config, predictor, retriever, rca_predictor, cmms, rul_predictor)
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
    try:
        await loop.run_in_executor(None, rul_predictor.initialize, MLFLOW_URI, DATA_PATH)
    except Exception as e:
        print(f"[RULPredictor] No se pudo inicializar: {e}. Predicción RUL no estará disponible.")
    yield


app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.5.0",
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
        "version":             "0.5.0",
        "predictor_ready":     predictor.initialized,
        "rca_predictor_ready": rca_predictor.initialized,
        "rul_predictor_ready": rul_predictor.initialized,
        "langfuse_enabled":    _LANGFUSE_ENABLED,
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
    lf_trace = (
        _langfuse.trace(name="amia.process_input", input={"query": user_input.query, "session_id": user_input.session_id})
        if _langfuse
        else None
    )
    try:
        result = await graph.ainvoke({
            "query":                user_input.query,
            "conversation_history": history,
            "retrieved_docs":       [],
            "sensor_analysis":      None,
            "rul_prediction":       None,
            "economic_impact":      None,
            "work_order":           None,
            "next_agent":           "",
            "final_response":       "",
            "sources":              [],
        })
    except Exception as e:
        if lf_trace:
            lf_trace.update(level="ERROR", status_message=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    agent_used = result.get("next_agent", "unknown")
    response   = result["final_response"]
    sources    = result["sources"]

    if lf_trace:
        lf_trace.update(output={"agent_used": agent_used, "response": response[:500]})

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


@app.get("/work-orders")
async def list_work_orders() -> list[dict]:
    """Devuelve todas las órdenes de trabajo (abiertas y completadas)."""
    return cmms.list_orders()


@app.get("/work-orders/{wo_id}")
async def get_work_order(wo_id: str) -> dict:
    """Devuelve una orden de trabajo por su ID."""
    order = cmms.get_order(wo_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Orden {wo_id} no encontrada")
    return order


@app.patch("/work-orders/{wo_id}/complete")
async def complete_work_order(wo_id: str) -> dict:
    """Marca una orden de trabajo como completada."""
    order = cmms.get_order(wo_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Orden {wo_id} no encontrada")
    order["status"] = "completed"
    return order


@app.post("/predict/rul")
async def predict_rul(machine_id: str) -> dict:
    """Predice la vida útil restante de una máquina."""
    if not rul_predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor RUL no está disponible.")
    try:
        return rul_predictor.predict(machine_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predict/rul/all")
async def predict_rul_all() -> list[dict]:
    """Devuelve predicciones de RUL para todas las máquinas."""
    if not rul_predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor RUL no está disponible.")
    return rul_predictor.predict_all()


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
