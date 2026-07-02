# Changelog

All notable changes to AMIA are documented here.

---

## [Unreleased] — AMIA Platform v2

### Etapa 0 — Fundaciones

#### 0.1 PostgreSQL + Alembic *(en progreso)*
- SQLAlchemy 2.0 async models: todas las tablas del dominio v2
- Alembic con migración inicial
- Fixtures pytest con base de datos efímera

---

## [0.6.0] — AMIA v1 Production & Portfolio Polish

### Added
- Rate limiting Redis (sliding window, 10 req/min, HTTP 429 + Retry-After)
- `/metrics/kpis` — KPIs ejecutivos: máquinas, alertas, riesgo económico, RUL promedio
- `/evaluate` — suite de evaluación formal: AUC-ROC, F1, Accuracy, Top-3, RMSE, R²
- `/metrics/drift` — detección de data drift con Evidently AI
- Executive Summary tiles en el dashboard Next.js
- Dockerfiles multi-stage (backend + frontend) + docker-compose.prod.yml

### Fixed
- evaluate.py: usar `mlflow.xgboost.load_model`, importar `build_features` de cada script de entrenamiento
- evaluate.py: filtro RCA usa `rca_label.notna()` (solo las 5 clases reales, no "normal")
- monitor_drift.py: importar desde `evidently.legacy` (API 0.7.x); target `is_failure`

## [0.5.0] — AMIA v1 RUL + Langfuse

### Added
- Modelo XGBoost RUL (RMSE ~118h, R² 0.34); registrado en MLflow como `amia-rul-model`
- Nodo `rul_analyst` en LangGraph; routing dual (directo + cadena de alerta)
- Endpoints `/predict/rul` y `/predict/rul/all`
- Sección "Vida Útil Restante" en dashboard (RULCard, barra de progreso por urgencia)
- Integración Langfuse (trazas LLM con `LangfuseCallbackHandler`)

## [0.4.0] — AMIA v1 Economic Impact + CMMS

### Added
- Agente `economic_analyst`: coste de parada, riesgo económico por tipo de máquina
- CMMS mock: gestión de órdenes de trabajo (`/work-orders`)
- Panel de órdenes de trabajo en dashboard

## [0.3.0] — AMIA v1 RCA

### Added
- Modelo XGBoost multiclase RCA (5 clases: bearing_wear, cavitation, electrical_failure, misalignment, overheating)
- Agente `sensor_analyst` con diagnóstico automático de causa raíz

## [0.2.0] — AMIA v1 RAG + Multi-agent

### Added
- RAG con Qdrant + CrossEncoder re-ranking
- LangGraph Supervisor + 5 agentes especializados
- Failure prediction XGBoost (AUC ~0.996)

## [0.1.0] — AMIA v1 Base

### Added
- FastAPI backend, Next.js 14 frontend
- Dataset sintético (5 máquinas, 43,800 lecturas)
- Pipeline de datos y entrenamiento ML
