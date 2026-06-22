# AMIA — Roadmap de Desarrollo Incremental

Cada versión es independientemente ejecutable y demostrable. El orden importa: cada versión asienta la infraestructura que usa la siguiente.

---

## Mapa general

| Versión | Nombre | Semanas est. | Estado |
|---------|--------|-------------|--------|
| **V0** | Fundación + Datos Sintéticos | 1-2 | ✅ Completo |
| **V1** | RAG + Chat Agent + Redis Cache | 2-3 | ✅ Completo |
| **V2** | Failure Prediction + Semantic Cache | 2-3 | 🔲 Pendiente |
| **V3** | Root Cause Analysis + Economic Impact | 2-3 | 🔲 Pendiente |
| **V4** | RUL + Observabilidad | 3-4 | 🔲 Pendiente |
| **V5** | Producción & Portfolio Polish | 2-3 | 🔲 Pendiente |

---

## V0 — Fundación + Datos Sintéticos ✅

**Objetivo**: Esqueleto del monorepo listo para construir encima. Dataset sintético industrial que alimenta todos los modelos ML futuros.

**Introduce**:
- Monorepo Python con uv workspaces (`backend/`, `ml/`, `shared/`)
- Schemas Pydantic compartidos (`amia_shared`)
- Docker Compose: PostgreSQL + Qdrant + Redis
- Generador de datos sintéticos con física de fallos realista
- GitHub Actions CI (lint, type-check, smoke test del generador)

**Entregable**:
```bash
./scripts/demo_v0.sh
# → genera 43,800 filas, 5 máquinas, 34 eventos de fallo
# → 8 plots EDA en data/synthetic/plots/
```

**Artefactos clave**:
- `data/synthetic/sensor_readings.parquet` — dataset principal (no cambiar schema en versiones futuras)
- `ml/amia_ml/synthetic/generator.py` — `AMIADataGenerator`
- `shared/amia_shared/schemas.py` — `MachineType`, `FailureMode`, `SensorReading`, `FailureEvent`
- `infra/docker-compose.yml` — servicios locales

---

## V1 — RAG + Chat Agent + Redis Cache ✅

**Objetivo**: Agente que responde preguntas sobre manuales y SOPs industriales, citando fuentes, con historial de conversación persistente y caché de respuestas para reducir latencia y costos de API.

**Implementado**:
- Pipeline de ingestión: PDF/MD/TXT → chunks → embeddings → Qdrant
- RAG multi-agente con LangGraph: Supervisor (Haiku) → DocExpert → Synthesizer (Sonnet)
- Re-ranking semántico con CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- API REST: `POST /ingest`, `POST /process_input`, `GET /health`
- **Historial de conversación** por sesión en Redis (ventana 10 turnos, TTL 24h)
- **Exact Cache** — embeddings y respuestas LLM cacheadas por hash exacto del input
  - Embeddings: `hash(texto) → vector float[384]` — evita recomputar con SentenceTransformer
  - Respuestas LLM: `hash(query + context_hash) → respuesta` — latencia < 1ms vs ~3s
  - TTL configurable por tipo de dato (`cache_ttl` en `RAGConfig`)
- Frontend Next.js básico: chat con renderizado Markdown, badge agente, fuentes colapsables

**Stack nuevo**:
- `anthropic` SDK + `langgraph` — agente multi-nodo
- `sentence-transformers` — embeddings locales (all-MiniLM-L6-v2, 384 dims)
- `qdrant-client` — vector DB con búsqueda coseno
- `pymupdf` — parsing de PDFs
- `redis` — historial de sesión + exact cache
- Next.js 14 + Tailwind + `react-markdown` — frontend

**Entregable**:
```bash
docker compose -f infra/docker-compose.yml up -d
uv run uvicorn app.main:app --app-dir backend --port 8000
cd frontend && npm run dev      # localhost:3000
# → Agente responde con fuentes citadas
# → Segunda pregunta del mismo turno: respuesta desde caché (~1ms)
# → Historial multi-turno: "y cómo se detecta ese fallo?" → resuelve contexto
```

---

## V2 — Failure Prediction + Semantic Cache

**Objetivo**: El agente predice riesgo de fallo en tiempo real. Dashboard con indicadores por máquina. Redis Semantic Cache para consultas similares.

**Introduce**:

### ML — Failure Prediction
- Feature engineering sobre `sensor_readings.parquet` (V0): RMS, std, kurtosis, tendencia en ventanas de 1h/8h/24h
- Modelo XGBoost/LightGBM: clasificación binaria (fallo en próximas 24h)
- MLflow: experiment tracking + Model Registry (versiones del modelo)
- Endpoint: `POST /predict/failure` → `{machine_id, risk_score, failure_probability}`
- Tool del agente: `predict_failure_risk(machine_id)` — el Supervisor la activa cuando detecta un `machine_id` en la query
- Simulador de sensores: script que publica lecturas al backend cada N segundos
- Dashboard: tarjetas por máquina con semáforo verde/amarillo/rojo en tiempo real

### Redis — Semantic Cache
- **Semantic Cache**: antes de llamar al LLM, busca queries similares en Redis usando embeddings
  - Si similitud coseno > 0.95 con una query cacheada → devuelve la respuesta almacenada
  - Implementación: `redis-py` con `RedisSearch` o búsqueda vectorial en Qdrant (colección `query_cache`)
  - Diferencia clave con Exact Cache: "¿qué es el desgaste?" y "explícame el desgaste" → misma respuesta cacheada
  - TTL más corto (1-2h) porque el contexto puede cambiar con nuevas ingestas
- **Cache hit ratio** visible en logs y en `/health` extendido

**Stack nuevo**:
- `mlflow` — experiment tracking + model registry (nuevo servicio en Docker Compose)
- `xgboost` / `lightgbm` — modelos de clasificación
- `scikit-learn` — pipelines de feature engineering
- WebSockets o polling SSE para streaming de sensores al dashboard

**Decisión de diseño**: el Semantic Cache se implementa como middleware entre FastAPI y LangGraph — intercepta antes de `graph.ainvoke()` y guarda después. No modifica los nodos del grafo.

**Entregable**:
```bash
# Descomentar mlflow en infra/docker-compose.yml
docker compose up -d
python ml/amia_ml/train_failure_prediction.py  # entrena + registra en MLflow
# → Dashboard: 5 máquinas en tiempo real
# → Una máquina entra en riesgo alto → agente genera alerta automática
# → Segunda consulta idéntica o similar → respuesta desde Semantic Cache
# → /health devuelve cache_hit_ratio: 0.43
```

---

## V3 — Root Cause Analysis + Economic Impact

**Objetivo**: El agente diagnostica la causa del fallo, estima el costo económico y crea una orden de trabajo.

**Introduce**:
- Modelo XGBoost multiclase para RCA (bearing_wear, misalignment, electrical_failure, etc.)
- Módulo `economic_impact.py`: OEE × producción × margen → pérdida estimada por hora
- CMMS mock: servicio que simula creación de órdenes de trabajo
- Tools del agente: `analyze_root_cause(machine_id)`, `calculate_economic_impact(machine_id, hours)`, `create_work_order(...)`
- Razonamiento multi-step: alarma → diagnóstico → costo → orden con SOP adjunto
- Panel en dashboard: "Impacto económico evitado" acumulado

**Stack nuevo**:
- CMMS mock como microservicio FastAPI separado (o módulo en el mismo backend)

**Entregable**:
```
alarma detectada → agente razona en 3-4 pasos →
orden de trabajo creada con: diagnóstico + costo estimado ($X/h) + SOP adjunto
```

---

## V4 — RUL + LangGraph + Observabilidad

**Objetivo**: Predicción de vida útil restante. Arquitectura agentic robusta con grafo de estados. Trazabilidad completa de decisiones.

**Introduce**:
- Modelo LSTM / Temporal Fusion Transformer (TFT) para RUL (horas restantes ± intervalo de confianza)
- **LangGraph** — reemplaza el agente lineal por un grafo de estados: `Analyze → Plan → Execute → Summarize`
- **Langfuse** — observabilidad LLM: traces, latencia por tool, costo de tokens por sesión
- Sistema de alertas: si RUL < umbral configurable → notificación automática
- Tool del agente: `predict_rul(machine_id)` → "340 horas ± 40"
- Dashboard: gráfica de RUL por máquina con banda de confianza

**Stack nuevo**:
- `torch` + `pytorch-lightning` — LSTM/TFT
- `langgraph` — orquestación del agente
- `langfuse` — observabilidad LLM (nuevo servicio en Docker Compose)

**Entregable**:
```bash
# Docker Compose: descomentar langfuse en docker-compose.yml
# → Langfuse dashboard mostrando trace completo de una sesión con 4+ tool calls
# → Dashboard de RUL con alertas automáticas
```

---

## V5 — Producción & Portfolio Polish

**Objetivo**: Versión lista para mostrar en entrevistas. Métricas formales, multi-agent, executive dashboard, hardening de producción.

**Introduce**:
- Multi-agent con LangGraph: coordinador + agentes especializados (sensor analyst, doc expert, economic analyst)
- Suite de evaluación formal: retrieval precision@k, agent task completion rate, AUC/RMSE de modelos
- **Executive dashboard**: costos evitados, MTTR reducido, ahorro anual estimado
- **Evidently AI** — ML monitoring: data drift en producción

### Redis — Rate Limiting y hardening de producción
- **Rate Limiting por usuario**: máx. N requests/minuto por `session_id` o IP usando contador Redis con TTL sliding window
  - Protege contra abuso de API y evita exceder límites de Anthropic
  - Middleware FastAPI: si contador > umbral → HTTP 429 con `Retry-After`
  - Configurable por tier: usuarios anónimos vs. autenticados
- **Persistencia de sesión entre pods**: en arquitecturas Kubernetes, Redis actúa como store compartido entre réplicas del backend — el `session_id` funciona independientemente del pod que sirva la request
- **Cache warming**: script que pre-cachea queries frecuentes al arrancar el sistema

- Dockerfiles multi-stage optimizados + `docker-compose.prod.yml`
- README con guía de inicio en 5 minutos + demo video

**Stack nuevo**:
- `evidently` — ML monitoring y data drift
- `slowapi` o middleware custom — rate limiting sobre Redis
- Docker multi-stage builds
- (Opcional) Azure deployment: AKS + Azure OpenAI

**Entregable**:
```bash
./scripts/demo_v5.sh
# → executive dashboard con KPIs
# → evaluation report con métricas formales
# → rate limiting: >10 req/min → 429 con Retry-After
# → suite de evaluación end-to-end
```

---

## Stack tecnológico completo

| Capa | V1 | V2 | V3 | V4 | V5 |
|------|----|----|----|----|-----|
| LLM | claude-sonnet-4-6 | ← | ← | ← | ← |
| Agente | LangGraph (Supervisor + Specialists) | ← | ← | ← | Multi-agent |
| RAG | Qdrant + CrossEncoder | ← | ← | ← | ← |
| Embeddings | all-MiniLM-L6-v2 | ← | ← | ← | ← |
| Modelos ML | — | XGBoost/LightGBM | + XGB multiclase | + LSTM/TFT | ← |
| Experiment tracking | — | MLflow | ← | ← | ← |
| LLM observability | — | — | — | Langfuse | ← |
| ML monitoring | — | — | — | — | Evidently AI |
| Redis | Historial sesión + Exact Cache | + Semantic Cache | ← | ← | + Rate Limiting |
| Backend | FastAPI | ← | ← | ← | ← |
| Frontend | Next.js + Tailwind | ← | ← | ← | ← |
| Vector DB | Qdrant | ← | ← | ← | ← |
| Infra local | Docker Compose | + MLflow | ← | + Langfuse | prod config |

---

## Principios de desarrollo

- **No romper el schema de datos**: `sensor_readings.parquet` de V0 es el contrato de datos. Los modelos de V2-V4 lo consumen directamente.
- **Cada versión tiene su demo script**: `scripts/demo_vX.sh` — levanta el sistema y muestra el flujo crítico.
- **Git tags por versión**: `git tag v0`, `v1`, etc. al completar cada versión.
- **Primero funcional, luego optimizado**: no sobreingeniear en versiones tempranas.
- **Smoke tests por versión**: un test de integración que cubre el flujo crítico de esa versión.
