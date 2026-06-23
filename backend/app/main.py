import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # carga .env antes de que los agentes lean ANTHROPIC_API_KEY

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .graph import get_graph
from .models import FailurePredictionResponse, InputQuery
from .services.conversation import ConversationStore
from .services.ingestion import IngestionPipeline, parse_document
from .services.predictor import FailurePredictor
from .services.rag_config import RAGConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
DATA_PATH  = Path(os.getenv("DATA_PATH", REPO_ROOT / "data/synthetic/sensor_readings.parquet"))

config    = RAGConfig()
pipeline  = IngestionPipeline(config)
predictor = FailurePredictor()
graph     = get_graph(config, predictor)   # predictor se inicializa después en lifespan
store     = ConversationStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar predictor en un thread pool para no bloquear el event loop
    # (la construcción de features es CPU-bound y tarda ~30s)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, predictor.initialize, MLFLOW_URI, DATA_PATH)
    except Exception as e:
        print(f"[Predictor] No se pudo inicializar: {e}. El endpoint /predict/failure no estará disponible.")
    yield


app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.2.0",
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
    return {
        "status": "ok",
        "version": "0.2.0",
        "predictor_ready": predictor.initialized,
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """
    Recibe un archivo (PDF, Markdown, TXT), lo parsea, genera embeddings
    y almacena los chunks en Qdrant.
    """
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


@app.post("/predict/failure", response_model=FailurePredictionResponse)
async def predict_failure(machine_id: str) -> dict:
    """
    Predice si una máquina fallará en las próximas 24 horas.

    Devuelve la probabilidad de fallo, un nivel de alerta (green/yellow/red)
    y si supera el umbral óptimo calibrado durante el entrenamiento.
    """
    if not predictor.initialized:
        raise HTTPException(
            status_code=503,
            detail="El predictor no está disponible. Verifica que el modelo fue entrenado con train_failure_prediction.py.",
        )
    try:
        result = predictor.predict(machine_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return result


@app.get("/predict/failure/all")
async def predict_all_machines() -> list[dict]:
    """Devuelve predicciones para todas las máquinas conocidas."""
    if not predictor.initialized:
        raise HTTPException(status_code=503, detail="El predictor no está disponible.")
    return predictor.predict_all()


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
