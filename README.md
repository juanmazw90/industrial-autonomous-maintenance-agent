# AMIA — Autonomous Maintenance Intelligence Agent

Sistema multi-agente de mantenimiento predictivo industrial que combina RAG, ML y LangGraph para predecir fallos, diagnosticar causas raíz, estimar vida útil restante (RUL) e impacto económico en tiempo real.

## Stack

| Capa | Tecnología |
|---|---|
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Agentes | LangGraph (Supervisor + 5 agentes especializados) |
| RAG | Qdrant + CrossEncoder re-ranking |
| ML | XGBoost (failure, RCA multiclase, RUL) + MLflow |
| Observabilidad | Langfuse (trazas LLM) + Evidently AI (data drift) |
| Cache | Redis (historial sesión + semantic cache) |
| Backend | FastAPI 0.6.0 |
| Frontend | Next.js 14 + Tailwind CSS |
| Infra | Docker Compose (local) / multi-stage (producción) |

## Quickstart (< 5 minutos)

### Prerequisitos

- Docker Desktop corriendo
- Python 3.12+ con [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- Clave de API de Anthropic

### 1. Variables de entorno

```bash
cp .env.example .env
# Editar .env y añadir ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Levantar infraestructura

```bash
docker compose -f infra/docker-compose.yml up -d
```

### 3. Generar datos y entrenar modelos

```bash
# Generar dataset sintético (43,800 lecturas, 5 máquinas, ~34 eventos de fallo)
uv run python ml/amia_ml/synthetic/generator.py

# Entrenar los 3 modelos ML (registrados en MLflow)
uv run python ml/amia_ml/train_failure_prediction.py
uv run python ml/amia_ml/train_rca.py
uv run python ml/amia_ml/train_rul.py
```

### 4. Iniciar backend

```bash
uv run uvicorn app.main:app --app-dir backend --port 8000
# → http://localhost:8000/health  {"status":"ok","version":"0.6.0","predictor_ready":true,...}
```

### 5. Iniciar frontend

```bash
cd frontend && npm install && npm run dev
# → http://localhost:3000/dashboard
```

---

## API Reference

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del sistema y versión |
| `/process_input` | POST | Chat multi-agente (RAG, sensor, RUL, económico) |
| `/predict/failure/all` | GET | Predicción de fallo 24h para todas las máquinas |
| `/predict/rul/all` | GET | Vida útil restante (RUL) para todas las máquinas |
| `/sensors/reading` | POST | Ingesta lectura de sensor en tiempo real |
| `/work-orders` | GET | Listar órdenes de trabajo |
| `/metrics/kpis` | GET | KPIs ejecutivos: alertas, riesgo económico, RUL flota |
| `/evaluate` | GET | Suite de evaluación formal (AUC, RMSE, Accuracy) |
| `/metrics/drift` | GET | Detección de data drift con Evidently AI |
| `/ingest` | POST | Ingestión de documentos PDF/MD en Qdrant |

### Rate Limiting

Los endpoints `/process_input` y `/predict/*` tienen límite de **10 requests/minuto por IP** (configurable via `RATE_LIMIT_REQUESTS` y `RATE_LIMIT_WINDOW_SECONDS`). Al excederse: `HTTP 429` con cabecera `Retry-After`.

---

## Arquitectura de agentes

```
User Query
    │
    ▼
Supervisor (claude-haiku) ── route ──►  doc_expert    ──► synthesizer ──► User
                                  ├──► sensor_analyst ──► [verde] ──────► synthesizer
                                  │                   └──► rul_analyst ──► economic_analyst ──► WO creator ──► synthesizer
                                  ├──► rul_analyst ──────────────────────────────────────────────────────────► synthesizer
                                  └──► synthesizer
```

## Flujo de datos ML

```
sensor_readings.parquet
    ├── train_failure_prediction.py  → amia-failure-prediction (XGBoost, AUC ~0.95)
    ├── train_rca.py                 → amia-rca-model (XGBoost multiclase, Acc ~0.85)
    └── train_rul.py                 → amia-rul-model (XGBoost regressor, RMSE ~30h)
```

## Despliegue en producción

```bash
# Build y run con Docker multi-stage
docker compose -f infra/docker-compose.prod.yml --env-file .env.prod up -d
```

Ver [infra/docker-compose.prod.yml](infra/docker-compose.prod.yml) para variables de entorno requeridas.

---

## Evaluación formal

```bash
# Métricas sobre test split (20% temporal)
uv run python ml/amia_ml/evaluate.py
# → data/evaluation/evaluation_report.json

# Data drift monitoring con Evidently AI
uv run python ml/amia_ml/monitor_drift.py
# → data/drift_report.html  (reporte interactivo)
# → data/drift_summary.json
```

O vía API:
```bash
curl http://localhost:8000/evaluate        # JSON con métricas
curl http://localhost:8000/metrics/drift   # JSON resumen drift
```
