import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from pathlib import Path as _Path
from typing import Any

from dotenv import load_dotenv

# Ruta explícita: sube desde backend/app/ hasta la raíz del repo donde vive .env
load_dotenv(_Path(__file__).resolve().parents[2] / ".env", override=True)

# XGBoost debe inicializar su runtime OpenMP antes de que torch (importado vía
# sentence-transformers más abajo) cargue el suyo — el orden inverso segfaultea.
from .infra.native_libs import ensure_openmp_order

ensure_openmp_order()

import json

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .agents.context import set_event_loop as _set_instrumentation_loop
from .api.v2.agents import router as agents_router
from .api.v2.config import router as config_router
from .api.v2.events import router as events_router
from .api.v2.ml import router as ml_router
from .api.v2.operations import router as operations_router
from .api.v2.platform import router as platform_router
from .domain.alerting import start_alert_job
from .graph import get_graph
from .infra.db.base import get_db
from .infra.db.models import Machine, WorkOrder
from .infra.demo_identity import DemoIdentityMiddleware
from .infra.settings import settings
from .middleware.rate_limiter import RateLimitMiddleware
from .ml.explain import load_machine_registry
from .models import FailurePredictionResponse, IncomingSensorReading, InputQuery
from .observability.audit_middleware import AuditMiddleware
from .observability.correlation import CorrelationIdMiddleware
from .observability.logging import configure_logging
from .rag.metrics import InstrumentedRetriever
from .services.conversation import ConversationStore
from .services.feature_store import load_feature_bundle
from .services.ingestion import IngestionPipeline, parse_document
from .services.predictor import FailurePredictor
from .services.rag_config import RAGConfig
from .services.rca_predictor import RCAPredictor
from .services.retrieval import Retriever
from .services.rul_predictor import RULPredictor
from .services.semantic_cache import SemanticCache

# ── Langfuse (opcional — activo cuando LANGFUSE_PUBLIC_KEY está configurada) ──
try:
    from langfuse import Langfuse as _Langfuse
    from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
    _LANGFUSE_ENABLED = bool(settings.langfuse_public_key)
    _langfuse: "_Langfuse | None" = (
        _Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        if _LANGFUSE_ENABLED
        else None
    )
except Exception:
    _LANGFUSE_ENABLED = False
    _langfuse = None
    _LangfuseCallbackHandler = None  # type: ignore[assignment,misc]


configure_logging()

REPO_ROOT  = Path(__file__).resolve().parents[2]
MLFLOW_URI = settings.mlflow_tracking_uri
DATA_PATH  = Path(settings.data_path)

config    = RAGConfig()
retriever = InstrumentedRetriever(Retriever(config))
pipeline  = IngestionPipeline(config)
predictor     = FailurePredictor()
rca_predictor = RCAPredictor()
rul_predictor = RULPredictor()
graph         = get_graph(config, predictor, retriever, rca_predictor, rul_predictor)
store     = ConversationStore(settings.redis_url)
sem_cache = SemanticCache(retriever)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register the running loop so instrument_tool can fire DB writes from threads.
    loop = asyncio.get_running_loop()
    _set_instrumentation_loop(loop)

    # Load machine code→UUID mapping for ml/explain.py prediction persistence.
    await load_machine_registry()

    # Inicializar semantic cache (crea colección en Qdrant si no existe)
    try:
        await sem_cache.initialize()
    except Exception as e:
        print(f"[SemanticCache] No se pudo inicializar: {e}")

    # Features del histórico: se computan UNA vez y se comparten entre los 3 predictores
    bundle = None
    try:
        bundle = await loop.run_in_executor(None, load_feature_bundle, DATA_PATH)
    except Exception as e:
        print(f"[FeatureStore] No se pudo cargar el histórico: {e}. Predictores no disponibles.")

    if bundle is not None:
        try:
            await loop.run_in_executor(None, predictor.initialize, MLFLOW_URI, DATA_PATH, bundle)
        except Exception as e:
            print(f"[Predictor] No se pudo inicializar: {e}. /predict/failure no estará disponible.")
        try:
            await loop.run_in_executor(None, rca_predictor.initialize, MLFLOW_URI, DATA_PATH, bundle)
        except Exception as e:
            print(f"[RCAPredictor] No se pudo inicializar: {e}. Diagnóstico RCA no estará disponible.")
        try:
            await loop.run_in_executor(None, rul_predictor.initialize, MLFLOW_URI, DATA_PATH, bundle)
        except Exception as e:
            print(f"[RULPredictor] No se pudo inicializar: {e}. Predicción RUL no estará disponible.")

    # Start background alert evaluation job (runs every 60 s).
    _alert_task = start_alert_job()
    yield
    _alert_task.cancel()
    try:
        await _alert_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(DemoIdentityMiddleware)
app.add_middleware(AuditMiddleware)

app.include_router(events_router)
app.include_router(config_router)
app.include_router(operations_router)
app.include_router(agents_router)
app.include_router(ml_router)
app.include_router(platform_router)

Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=[r"/health", r"/docs", r"/openapi\.json", r"/redoc"],
).instrument(app).expose(app, endpoint="/prom-metrics", include_in_schema=False)

# Tag legacy v1 routes with HTTP deprecation headers (RFC 8594).
_V1_PREFIXES = ("/predict/", "/sensors/", "/work-orders", "/evaluate", "/metrics/")

class _DeprecationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if any(path.startswith(p) for p in _V1_PREFIXES):
                async def _send_with_headers(message):
                    if message["type"] == "http.response.start":
                        headers = list(message.get("headers", []))
                        headers.extend([
                            (b"deprecation", b"true"),
                            (b"sunset", b"Sat, 01 Jan 2027 00:00:00 GMT"),
                            (b"link", b'</api/v2>; rel="successor-version"'),
                        ])
                        message = {**message, "headers": headers}
                    await send(message)
                await self.app(scope, receive, _send_with_headers)
                return
        await self.app(scope, receive, send)

app.add_middleware(_DeprecationMiddleware)

_RATE_LIMIT  = settings.rate_limit_requests
_RATE_WINDOW = settings.rate_limit_window_seconds
app.add_middleware(
    RateLimitMiddleware, redis_url=settings.redis_url, limit=_RATE_LIMIT, window=_RATE_WINDOW
)


@app.get("/health")
async def health() -> dict:
    cache_stats = await sem_cache.stats()
    return {
        "status":              "ok",
        "version":             "0.7.0",
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

    loop = asyncio.get_running_loop()
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
    history = await store.get_history(user_input.session_id)

    # ── Semantic cache check ──────────────────────────────────────────────
    cached = await sem_cache.get(user_input.query)
    if cached:
        # Hit: devolver respuesta sin llamar a LangGraph
        await store.append_turn(user_input.session_id, user_input.query, cached["response"])
        return {
            "response":   cached["response"],
            "sources":    cached["sources"],
            "agent_used": cached["agent_used"],
            "session_id": user_input.session_id,
            "cached":     True,
            "similarity": cached.get("similarity"),
        }

    # ── LangGraph invocation ──────────────────────────────────────────────
    lf_callbacks = [_LangfuseCallbackHandler()] if (_langfuse and _LangfuseCallbackHandler is not None) else []
    try:
        result = await graph.ainvoke(
            {
                "session_id":           user_input.session_id,
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
            },
            config={"callbacks": lf_callbacks},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    agent_used = result.get("next_agent", "unknown")
    response   = result["final_response"]
    sources    = result["sources"]

    await store.append_turn(user_input.session_id, user_input.query, response)

    # ── Store in semantic cache (solo doc_expert) ─────────────────────────
    await sem_cache.set(user_input.query, response, sources, agent_used)

    return {
        "response":   response,
        "sources":    sources,
        "agent_used": agent_used,
        "session_id": user_input.session_id,
        "cached":     False,
    }


@app.post("/process_input/stream")
async def process_input_stream(user_input: InputQuery) -> StreamingResponse:
    """Server-Sent Events endpoint — streams synthesizer tokens as they are generated."""
    history = await store.get_history(user_input.session_id)

    async def event_gen():
        # ── Semantic cache hit ────────────────────────────────────────────────
        cached = await sem_cache.get(user_input.query)
        if cached:
            await store.append_turn(user_input.session_id, user_input.query, cached["response"])
            # Send cached text in small chunks with delay for animation effect
            text = cached["response"]
            for i in range(0, len(text), 6):
                yield f"data: {json.dumps({'type': 'token', 'content': text[i:i+6]})}\n\n"
                await asyncio.sleep(0.018)
            done_evt = {
                "type": "done", "agent_used": cached["agent_used"], "sources": cached["sources"],
                "cached": True, "session_id": user_input.session_id,
            }
            yield f"data: {json.dumps(done_evt)}\n\n"
            return

        # ── LangGraph astream_events ──────────────────────────────────────────
        lf_callbacks = [_LangfuseCallbackHandler()] if (_langfuse and _LangfuseCallbackHandler is not None) else []

        accumulated = ""
        agent_used  = "synthesizer"
        sources: list = []

        try:
            async for event in graph.astream_events(
                {
                    "session_id":           user_input.session_id,
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
                },
                version="v2",
                config={"callbacks": lf_callbacks},
            ):
                kind = event["event"]

                if kind == "on_custom_event" and event.get("name") == "synthesizer_token":
                    # Tokens emitidos por el synthesizer vía adispatch_custom_event
                    content = event.get("data", {}).get("content", "")
                    if content:
                        accumulated += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if output.get("next_agent"):
                            agent_used = output["next_agent"]
                        if output.get("sources"):
                            sources = output["sources"]
                        # Fallback: LangGraph didn't stream — animate the final response
                        if output.get("final_response") and not accumulated:
                            accumulated = output["final_response"]
                            for i in range(0, len(accumulated), 6):
                                yield f"data: {json.dumps({'type': 'token', 'content': accumulated[i:i+6]})}\n\n"
                                await asyncio.sleep(0.018)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        await store.append_turn(user_input.session_id, user_input.query, accumulated)
        await sem_cache.set(user_input.query, accumulated, sources, agent_used)

        done_evt = {
            "type": "done", "agent_used": agent_used, "sources": sources,
            "cached": False, "session_id": user_input.session_id,
        }
        yield f"data: {json.dumps(done_evt)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, predictor.update_with_reading, reading.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _wo_to_v1(wo: WorkOrder, machine_code: str | None) -> dict:
    return {
        "work_order_id":  wo.id,
        "machine_id":     machine_code or wo.machine_id,
        "title":          wo.title,
        "priority":       wo.priority,
        "estimated_cost": wo.estimated_cost or 0.0,
        "status":         wo.status,
        "created_at":     wo.created_at.isoformat(),
    }


@app.get("/work-orders")
async def list_work_orders(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Devuelve todas las órdenes de trabajo (abiertas y completadas)."""
    rows = (await db.execute(
        select(WorkOrder, Machine)
        .join(Machine, WorkOrder.machine_id == Machine.id)
        .order_by(WorkOrder.created_at.desc())
    )).all()
    return [_wo_to_v1(wo, m.code) for wo, m in rows]


@app.get("/work-orders/{wo_id}")
async def get_work_order(wo_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Devuelve una orden de trabajo por su ID."""
    row = (await db.execute(
        select(WorkOrder, Machine)
        .join(Machine, WorkOrder.machine_id == Machine.id)
        .where(WorkOrder.id == wo_id)
    )).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail=f"Orden {wo_id} no encontrada")
    wo, m = row
    return _wo_to_v1(wo, m.code)


@app.patch("/work-orders/{wo_id}/complete")
async def complete_work_order(wo_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Marca una orden de trabajo como completada."""
    wo = (await db.execute(
        select(WorkOrder).where(WorkOrder.id == wo_id)
    )).scalar_one_or_none()
    if not wo:
        raise HTTPException(status_code=404, detail=f"Orden {wo_id} no encontrada")
    wo.status = "completed"
    await db.commit()
    await db.refresh(wo)
    return _wo_to_v1(wo, None)


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


@app.get("/evaluate")
async def evaluate_models() -> dict:
    """
    Ejecuta la suite de evaluación formal sobre el test split (20% temporal).
    Devuelve AUC/F1/Precision/Recall para failure prediction, Accuracy/Top-3/F1-macro
    para RCA, y RMSE/MAE/R² para RUL.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "ml"))
    try:
        from amia_ml.evaluate import run_all
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, run_all)
        return {"status": "ok", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {e}")


@app.get("/metrics/drift")
async def get_drift_report() -> dict:
    """
    Ejecuta Evidently AI para detectar data drift entre el 60% de referencia
    y el 20% más reciente del dataset. Genera data/drift_report.html.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "ml"))
    try:
        from amia_ml.monitor_drift import run_drift_report
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, run_drift_report)
        return {"status": "ok", **summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en drift monitoring: {e}")


@app.get("/metrics/kpis")
async def get_kpis(db: AsyncSession = Depends(get_db)) -> dict:
    """
    KPIs ejecutivos del sistema: estado de la flota, riesgo económico y degradación media.
    Usado por el executive dashboard del frontend.
    """
    failure_preds: list[dict[str, Any]] = predictor.predict_all() if predictor.initialized else []
    rul_preds:     list[dict[str, Any]] = rul_predictor.predict_all() if rul_predictor.initialized else []

    total_wo = (await db.execute(select(func.count(WorkOrder.id)))).scalar() or 0
    open_orders = (await db.execute(
        select(WorkOrder).where(WorkOrder.status == "open")
    )).scalars().all()

    alert_counts = {"green": 0, "yellow": 0, "red": 0}
    for p in failure_preds:
        level = p.get("alert_level", "green")
        alert_counts[level] = alert_counts.get(level, 0) + 1

    total_risk_usd = sum(o.estimated_cost or 0.0 for o in open_orders)

    avg_rul   = round(sum(r["hours_remaining"] for r in rul_preds) / len(rul_preds), 1) if rul_preds else None
    avg_degr = (
        round(sum(r["degradation_fraction"] for r in rul_preds) / len(rul_preds) * 100, 1)
        if rul_preds else None
    )

    return {
        "version":                 "0.7.0",
        "machines_monitored":      len(failure_preds),
        "machines_green":          alert_counts["green"],
        "machines_warning":        alert_counts["yellow"],
        "machines_critical":       alert_counts["red"],
        "work_orders_open":        len(open_orders),
        "work_orders_total":       total_wo,
        "risk_exposure_usd":       round(total_risk_usd, 2),
        "avg_rul_hours":           avg_rul,
        "fleet_degradation_pct":   avg_degr,
        "rate_limit":              {"requests": _RATE_LIMIT, "window_seconds": _RATE_WINDOW},
    }


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
