"""
Etapa 2.3 — ML Center / XAI / RAG routers.

GET /api/v2/models                     — list MLflow registered models
GET /api/v2/models/{name}/metrics      — ROC, confusion matrix, feature importance
GET /api/v2/models/{name}/drift        — Evidently drift summary
GET /api/v2/predictions/{id}/explain   — SHAP features + similar cases
GET /api/v2/rag/summary                — docs, chunks, latency, similarity, miss rate
GET /api/v2/rag/documents              — paginated chunk list from Qdrant
GET /api/v2/rag/queries                — recent RAG queries
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.db.base import get_db
from app.infra.db.models import ModelPrediction, RagQuery
from app.infra.settings import settings

router = APIRouter(prefix="/api/v2", tags=["ml"])

REPO_ROOT = Path(__file__).resolve().parents[4]
MLFLOW_URI = settings.mlflow_tracking_uri


# ── /models ────────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """List MLflow registered models with latest version metadata."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        registered = client.search_registered_models()
        result = []
        for rm in registered:
            latest = rm.latest_versions[0] if rm.latest_versions else None
            result.append({
                "name": rm.name,
                "description": rm.description,
                "latest_version": latest.version if latest else None,
                "latest_stage": latest.current_stage if latest else None,
                "latest_run_id": latest.run_id if latest else None,
                "tags": dict(rm.tags) if rm.tags else {},
            })
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")


@router.get("/models/{name}/metrics")
async def get_model_metrics(name: str) -> dict[str, Any]:
    """Return metrics logged for the latest version of a registered model."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        versions = client.get_latest_versions(name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Model '{name}' has no versions")
        v = versions[0]
        run = client.get_run(v.run_id)
        return {
            "model": name,
            "version": v.version,
            "run_id": v.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow error: {exc}")


@router.get("/models/{name}/drift")
async def get_model_drift(name: str) -> dict[str, Any]:
    """Return Evidently drift summary from latest drift report."""
    drift_file = REPO_ROOT / "data" / "drift_summary.json"
    if not drift_file.exists():
        return {
            "model": name,
            "status": "no_report",
            "message": "Run GET /metrics/drift to generate an Evidently report first.",
        }
    import json
    try:
        summary = json.loads(drift_file.read_text())
        return {"model": name, **summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read drift summary: {exc}")


# ── /predictions/{id}/explain ──────────────────────────────────────────────────

@router.get("/predictions/{pred_id}/explain")
async def explain_prediction(pred_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """SHAP feature contributions + 3 most similar past predictions."""
    pred = (await db.execute(
        select(ModelPrediction).where(ModelPrediction.id == pred_id)
    )).scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction '{pred_id}' not found")

    # Similar predictions: same machine, same model, nearest probability
    similar = (await db.execute(
        select(ModelPrediction)
        .where(
            ModelPrediction.machine_id == pred.machine_id,
            ModelPrediction.model_name == pred.model_name,
            ModelPrediction.id != pred.id,
            ModelPrediction.probability.is_not(None),
        )
        .order_by(
            func.abs(ModelPrediction.probability - (pred.probability or 0.0))
        )
        .limit(3)
    )).scalars().all()

    return {
        "prediction_id": pred.id,
        "machine_id": pred.machine_id,
        "model_name": pred.model_name,
        "probability": pred.probability,
        "prediction": pred.prediction,
        "shap_top_features": pred.shap_top_features or [],
        "created_at": pred.created_at.isoformat(),
        "similar_predictions": [
            {
                "id": s.id,
                "probability": s.probability,
                "shap_top_features": s.shap_top_features or [],
                "created_at": s.created_at.isoformat(),
            }
            for s in similar
        ],
    }


# ── /rag/summary ───────────────────────────────────────────────────────────────

@router.get("/rag/summary")
async def rag_summary(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Aggregated RAG performance stats: doc count, chunk count, avg latency,
    avg similarity, cache miss rate.
    """
    agg = (await db.execute(
        select(
            func.count(RagQuery.id).label("total_queries"),
            func.avg(RagQuery.retrieval_latency_ms).label("avg_latency_ms"),
            func.avg(RagQuery.avg_similarity).label("avg_similarity"),
        )
    )).one()

    misses_row = (await db.execute(
        select(func.count(RagQuery.id)).where(RagQuery.hit == False)  # noqa: E712
    )).scalar() or 0

    total = agg.total_queries or 0
    misses = misses_row
    miss_rate = round(misses / total, 3) if total else 0.0

    # Qdrant collection info for chunk count
    chunk_count = None
    try:
        from qdrant_client import AsyncQdrantClient

        from app.services.rag_config import RAGConfig
        cfg = RAGConfig()
        qdrant_url = f"http://{cfg.qdrant_host}:{cfg.qdrant_port}"
        client = AsyncQdrantClient(url=qdrant_url)
        info = await client.get_collection(cfg.collection_name)
        chunk_count = info.points_count
        await client.close()
    except Exception:
        pass

    return {
        "total_queries": total,
        "avg_latency_ms": round(float(agg.avg_latency_ms or 0), 1),
        "avg_similarity": round(float(agg.avg_similarity or 0), 3),
        "miss_rate": miss_rate,
        "hits": total - misses,
        "misses": misses,
        "chunk_count": chunk_count,
    }


@router.get("/rag/documents")
async def list_rag_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, description="Filter by text in payload"),
) -> dict[str, Any]:
    """List document chunks stored in Qdrant."""
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchText

        from app.services.rag_config import RAGConfig

        cfg = RAGConfig()
        client = AsyncQdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

        scroll_filter = None
        if search:
            scroll_filter = Filter(
                must=[FieldCondition(key="text", match=MatchText(text=search))]
            )

        results, _next = await client.scroll(
            collection_name=cfg.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        await client.close()

        return {
            "limit": limit,
            "offset": offset,
            "documents": [
                {
                    "id": str(point.id),
                    "payload": point.payload,
                }
                for point in results
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant error: {exc}")


@router.get("/rag/queries")
async def list_rag_queries(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    hit: bool | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Recent RAG retrieval queries with performance metrics."""
    q = select(RagQuery).order_by(RagQuery.created_at.desc())
    if hit is not None:
        q = q.where(RagQuery.hit == hit)

    total = (await db.execute(
        select(func.count(RagQuery.id))
        .where(RagQuery.hit == hit) if hit is not None
        else select(func.count(RagQuery.id))
    )).scalar() or 0

    queries = (await db.execute(q.limit(limit).offset(offset))).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "queries": [
            {
                "id": rq.id,
                "agent_run_id": rq.agent_run_id,
                "query": rq.query,
                "top_k": rq.top_k,
                "retrieval_latency_ms": rq.retrieval_latency_ms,
                "avg_similarity": rq.avg_similarity,
                "hit": rq.hit,
                "sources": rq.sources,
                "created_at": rq.created_at.isoformat(),
            }
            for rq in queries
        ],
    }
