# AMIA Platform v2 — Backlog

Generado desde SDD v2.0.0, Sección 11–12. Estado actualizado por etapa completada.

Leyenda: `[ ]` pendiente · `[x]` completado · `[~]` en progreso

---

## ETAPA 0 — Fundaciones · MUST

### 0.1 PostgreSQL + Alembic
- [~] Añadir `alembic` a backend/pyproject.toml
- [~] Crear `backend/app/infra/db/base.py` — `AsyncEngine` + `AsyncSession` factory
- [~] Crear `backend/app/infra/db/models.py` — todos los modelos ORM (Sección 4.1 del SDD)
  - `demo_users`, `plants`, `lines`, `machines`
  - `agent_runs`, `tool_calls`, `rag_queries`, `agent_evaluations`, `model_predictions`
  - `alerts`, `work_orders`, `audit_log`, `timeline_events`
  - `platform_config`
- [~] `alembic init backend/alembic` + configurar `env.py` con async engine
- [~] Crear migración inicial (`alembic revision --autogenerate`)
- [~] `backend/tests/conftest.py` — fixtures con DB PostgreSQL efímera (`pytest-asyncio` + `anyio`)
- [~] Smoke test: crear y consultar una `demo_user` via ORM

### 0.2 Identidad demo
- [ ] Seed de `demo_users` (5 roles: operator, supervisor, maintenance_manager, plant_director, ai_engineer)
- [ ] Middleware `X-Demo-User` → `request.state.actor` (resuelve contra `demo_users`)
- [ ] Sin header → actor `{"id": null, "name": "system", "role": "system"}`
- [ ] Test: request con/sin header resuelve actor correcto

### 0.3 Logging estructurado + correlation_id
- [ ] Añadir `structlog` a pyproject.toml
- [ ] Middleware que genera `correlation_id` (UUID) por request y lo propaga en headers + contextvars
- [ ] Configurar structlog con sink JSON a stdout + sink Postgres (tabla `logs` en `platform_config`)
- [ ] Todo el backend usa `log = structlog.get_logger()` (no `logging` crudo)
- [ ] Test: un request genera log estructurado con `correlation_id`

### 0.4 Event bus
- [ ] `backend/app/events/publisher.py` — `publish(event_type, payload, correlation_id)` vía Redis pub/sub `events:live`
- [ ] `backend/app/events/schemas.py` — Pydantic schemas de todos los eventos (Sección 5.3 del SDD)
- [ ] `GET /api/v2/events/stream` — SSE que consume `events:live` y reenvía al cliente
- [ ] Test: publicar evento → aparece en stream SSE (httpx + asyncio)

### 0.5 Seeder de demo
- [ ] `scripts/seed_demo.py` — escenario determinista (semilla fija):
  - Insertar 5 demo_users (un rol cada uno)
  - Insertar plant/lines/machines (Plant Alpha, 2 lines, 5 machines)
  - Machine 02 en estado crítico: prob. fallo 98%, RUL 18h
  - Timeline completo de Machine 02: anomalía → predicción → RCA → económico → WO
  - 3 alertas abiertas (1 critical, 1 high, 1 medium)
  - 2 work orders (1 open, 1 in_progress)
  - 10 agent_runs de ejemplo con tool_calls
  - 5 rag_queries de ejemplo
- [ ] `scripts/seed_demo.py --reset` limpia y re-siembra
- [ ] Test: ejecutar seed, verificar counts en DB

### 0.6 Reestructura backend
- [ ] Crear estructura objetivo:
  ```
  backend/app/
    api/          # routers FastAPI (thin controllers)
    domain/       # servicios de dominio
    infra/        # DB, Redis, Qdrant clients
    events/       # event bus + schemas
    observability/ # OTel, Prometheus, structlog config
    agents/       # LangGraph (sin cambio de topología)
    ml/           # carga de modelos, inferencia, SHAP
    rag/          # ingesta, retrieval, métricas
  ```
- [ ] Mover código v1 a nuevas rutas sin cambiar comportamiento
- [ ] Smoke tests: `GET /health`, `POST /process_input`, `GET /predict/failure/all` siguen funcionando
- [ ] Actualizar imports en todos los archivos afectados

---

## ETAPA 1 — Instrumentación de la capa de IA · MUST

### 1.1 Instrumentación LangGraph
- [ ] `backend/app/agents/instrumentation.py` — decorador `instrument_agent(name)`
  - Crea `agent_run` (estado `running`) con `parent_run_id`
  - Registra `tool_call` por cada tool invocada (input/output/latencia)
  - Al finalizar: tokens/coste, `reasoning_summary`, estado `success|error`
  - Publica evento `agent_run.finished` en Redis
- [ ] Envolver todos los nodos LangGraph existentes con `instrument_agent`
- [ ] Exportar span OTel con `traceparent = correlation_id`
- [ ] CA: toda ejecución de `/process_input` genera `agent_run` + `tool_calls` en DB

### 1.2 Instrumentación RAG + predicciones
- [ ] `backend/app/rag/metrics.py` — persiste `rag_queries` (latencia, similitud, hit/miss)
- [ ] `backend/app/ml/explain.py` — SHAP TreeExplainer; persiste en `model_predictions.shap_top_features`
- [ ] SHAP en cola background (no bloquea respuesta)
- [ ] CA: `GET /api/v2/rag/queries` devuelve últimas 20 consultas con métricas

### 1.3 Motor de alertas
- [ ] `backend/app/domain/alerting.py` — evalúa predicciones contra `platform_config.thresholds`
- [ ] Job periódico (cada 60s) que evalúa todas las máquinas y crea alertas si umbral superado
- [ ] On-prediction hook: al guardar `model_predictions`, evalúa inmediatamente
- [ ] Transiciones de estado: `new → acknowledged → assigned → resolved`
- [ ] CA: predicción >95% → alerta `severity=critical` creada en <5s

### 1.4 Auditoría + timeline
- [ ] `backend/app/domain/audit.py` — middleware captura toda mutación (POST/PATCH/DELETE)
- [ ] `domain/timeline.py` — inserta `timeline_events` desde agent decisions, predicciones, alertas, WOs
- [ ] `actor_type`: `user|agent|system`
- [ ] CA: crear WO vía API → `audit_log` registra actor + diff

### 1.5 Config store
- [ ] Tabla `platform_config` (key/value JSONB) con valores por defecto
- [ ] `backend/app/domain/config.py` — `get_config(key)` con cache Redis (TTL 60s, invalidación en PATCH)
- [ ] Valores por defecto: `thresholds.failure=0.85`, `thresholds.economic=5000`, `rag.top_k=5`, `llm.temperature=0.1`
- [ ] CA: cambiar threshold via `/api/v2/config` → motor de alertas usa el nuevo valor en <60s

---

## ETAPA 2 — API v2 · MUST

### 2.1 Routers Operations/Fleet/Assets
- [ ] `GET /api/v2/operations/summary` — KPIs ejecutivos agregados
- [ ] `GET /api/v2/fleet` — grid de máquinas con status + predicciones
- [ ] `GET /api/v2/assets/{machine_id}` — asset overview
- [ ] `GET /api/v2/assets/{machine_id}/timeline` — eventos filtrados
- [ ] `GET /api/v2/assets/{machine_id}/sensors` — series temporales (downsampled)

### 2.2 Routers Agents
- [ ] `GET /api/v2/agents/summary` — tarjetas por agente (p50/p95, calls, success, coste)
- [ ] `GET /api/v2/agents/runs` — listado con filtros
- [ ] `GET /api/v2/agents/runs/{run_id}` — detalle completo
- [ ] `GET /api/v2/agents/runs/{run_id}/trace` — árbol Tool Calls Explorer
- [ ] `GET/POST /api/v2/agents/evaluations` — resultados + lanzar batch LLM-as-judge

### 2.3 Routers Models/XAI/RAG
- [ ] `GET /api/v2/models` — modelos MLflow registrados
- [ ] `GET /api/v2/models/{name}/metrics` — ROC, CM, feature importance (desde artefactos JSON)
- [ ] `GET /api/v2/models/{name}/drift` — resumen Evidently + enlace HTML
- [ ] `GET /api/v2/predictions/{id}/explain` — SHAP + casos similares (kNN)
- [ ] `GET /api/v2/rag/summary` — métricas RAG agregadas
- [ ] `GET /api/v2/rag/documents` — inspección chunks Qdrant
- [ ] `GET /api/v2/rag/queries` — últimas consultas con métricas

### 2.4 Routers Operación de plataforma
- [ ] `GET/POST /api/v2/alerts` + `PATCH /api/v2/alerts/{id}` (transiciones)
- [ ] `GET/POST/PATCH /api/v2/work-orders`
- [ ] `GET /api/v2/logs` — filtros por service/agent/severity/correlation_id
- [ ] `GET /api/v2/audit` — filtros por actor/entidad
- [ ] `GET /api/v2/monitoring/services` — health checks activos
- [ ] `GET/PATCH /api/v2/config`
- [ ] `GET /api/v2/events/stream` (SSE — ya en 0.4)
- [ ] `GET /metrics` — Prometheus instrumentator

### 2.5 OpenAPI + cliente TypeScript
- [ ] OpenAPI spec limpia, `response_model` en todos los endpoints
- [ ] Generar cliente TS con `openapi-typescript`

### 2.6 Tests de integración
- [ ] ≥80% cobertura en `api/` y `domain/`
- [ ] httpx + DB efímera (fixtures de 0.1)

---

## ETAPA 3 — Frontend núcleo · MUST

### 3.1 Layout
- [ ] Sidebar con 12 secciones del menú (SDD 7.2)
- [ ] Topbar: breadcrumb, LiveIndicator (SSE), DemoUserSwitcher, health dot
- [ ] Tema oscuro industrial; design tokens Tailwind
- [ ] Zustand store: `actor` (identidad demo), `connected` (SSE)

### 3.2 Dashboard + Fleet View
- [ ] W-01 Executive Operations Center (4 bloques desde `/operations/summary`)
- [ ] W-02 Fleet View — MachineCard grid con filtro por status

### 3.3 AI Control Center + Trace Explorer
- [ ] W-04 Agent cards (p50/p95, calls, success, coste, tokens)
- [ ] Tabla de runs con filtros
- [ ] W-05 RunTraceTree — árbol colapsable supervisor→agentes→tools con IO expandible

### 3.4 Asset Detail + Timeline
- [ ] W-03 Tabs: Overview / Sensors / Timeline / Work Orders / Explainability / Documents
- [ ] Timeline vertical con eventos del seeder

### 3.5 Alert Center
- [ ] W-09 Tabla con filtros; acciones: Acknowledge / Assign / Resolve
- [ ] Contador actualizado vía SSE

---

## ETAPA 4 — Frontend IA/ML/RAG · MUST

### 4.1 ML Center
- [ ] W-06 Lista modelos (MLflow); gráficos ROC, CM, feature importance (Recharts)
- [ ] Panel drift con verdict Evidently + enlace HTML

### 4.2 Explainable AI
- [ ] SHAP bar chart (top-8 features) integrado en Asset Detail y Alertas
- [ ] 3 casos históricos similares

### 4.3 RAG Dashboard
- [ ] W-08 Métricas: docs, chunks, latencias, similitud, miss rate, top sources
- [ ] Inspector de documentos (Sheet lateral con chunks)

### 4.4 Agent Evaluation
- [ ] W-07 Dashboard de métricas (groundedness, hallucination, accuracy)
- [ ] Botón "Run evaluation batch" → POST `/api/v2/agents/evaluations`
- [ ] `eval/golden_set.json` — 20 preguntas de referencia

### 4.5 Logs Explorer
- [ ] W-12 Tabla virtualizada con filtros combinables
- [ ] Click en `correlation_id` → filtra toda la traza transversal

---

## ETAPA 5 — Operación de plataforma · SHOULD

- [ ] **5.1** Work Orders Kanban 4 columnas (menú contextual, sin drag&drop)
- [ ] **5.2** Monitoring unificado (W-11) + Prometheus/Grafana provisioning + OTel Collector
- [ ] **5.3** Audit Trail UI (W-13)
- [ ] **5.4** Settings (W-14) — formularios LLM/RAG/Thresholds
- [ ] **5.5** Deprecar endpoints v1 duplicados; README v2; vídeo/GIFs de demo

---

## ETAPA 6 — Diferenciadores opcionales · NICE TO HAVE

- [ ] **6.1** Digital Twin ligero (árbol Plant→Line→Machine→Sensors)
- [ ] **6.2** Drag&drop Kanban (dnd-kit)
- [ ] **6.3** Chat contextual por máquina
- [ ] **6.4** Predictive calendar
- [ ] **6.5** PDF report generator
- [ ] **6.6** Scenario simulator
