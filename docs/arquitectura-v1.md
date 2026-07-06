# AMIA V1 — Arquitectura y flujo de datos

Tres diagramas complementarios para entender el sistema:

| Diagrama | Pregunta que responde |
|----------|----------------------|
| 1 — Componentes | ¿Qué piezas existen y cómo se conectan? |
| 2 — Secuencia con payloads | ¿Qué pasa exactamente y con qué datos? |
| 3 — Contratos de datos | ¿Cómo está definida cada estructura? |

---

## Diagrama 1 — Arquitectura de componentes

```mermaid
graph TB
    subgraph Browser["🌐 Browser  localhost:3000"]
        UI["React Chat UI\napp/page.tsx\n\nsession_id (UUID) en memoria React\nhistorial visual de mensajes"]
    end

    subgraph NextJS["⚡ Next.js 14"]
        PROXY["Rewrite /api/* → http://localhost:8000/*\nnext.config.js"]
    end

    subgraph FastAPI["🚀 FastAPI  :8000\nbackend/app/main.py"]
        EP1["POST /process_input"]
        EP2["POST /ingest"]
        EP3["GET /health"]
    end

    subgraph LangGraph["🔀 LangGraph  StateGraph\nbackend/app/graph.py"]
        SUP["Supervisor\nagents/supervisor.py\nClaude Haiku — routing"]
        DOC["DocExpert\nagents/doc_expert.py\nRAG retrieval"]
        SYN["Synthesizer\nagents/synthesizer.py\nClaude Sonnet — generación"]
    end

    subgraph Services["⚙️ Services  backend/app/services/"]
        CONV["ConversationStore\nconversation.py"]
        ING["IngestionPipeline\ningestion.py"]
        RET["Retriever\nretrieval.py\nSentenceTransformer + CrossEncoder"]
    end

    subgraph Infra["🗄️ Docker Compose  infra/"]
        QDRANT[("Qdrant  :6333\nvector DB\ncollection: documents")]
        REDIS[("Redis  :6379\nhistorial por sesión\nTTL: 24h")]
    end

    subgraph Anthropic["☁️ Anthropic API"]
        HAIKU["claude-haiku-4-5\nrouting (tool_use)"]
        SONNET["claude-sonnet-4-6\ngeneración de respuesta"]
    end

    UI -->|"POST /api/process_input\n{query, session_id}"| PROXY
    PROXY -->|proxy transparente| EP1
    EP1 <-->|get_history / append_turn| CONV
    CONV <-->|"GET/SETEX conv:{session_id}"| REDIS
    EP1 -->|"graph.ainvoke(AMIAState)"| LangGraph
    SUP -->|"messages.create(tool_use)"| HAIKU
    HAIKU -->|"next_agent: doc_expert\|synthesizer"| SUP
    SUP -.->|"condicional"| DOC
    SUP -.->|"condicional"| SYN
    DOC -->|"retrieve(query, top_k=20)"| RET
    RET -->|"embed query → search coseno"| QDRANT
    QDRANT -->|"20 chunks con vector + payload"| RET
    RET -->|"rerank → top 3"| DOC
    DOC -->|"retrieved_docs"| SYN
    SYN -->|"messages.create(system + history + context)"| SONNET
    SONNET -->|"TextBlock(respuesta)"| SYN
    EP2 -->|"parse → chunk → embed → upsert"| ING
    ING -->|"PointStruct(uuid, vector, payload)"| QDRANT
```

**Lectura clave:**
- El proxy de Next.js es invisible para el usuario: `/api/process_input` → `:8000/process_input`
- LangGraph es el orquestador: decide qué nodo ejecutar en cada paso
- Qdrant y Redis son los únicos componentes con estado persistente
- La ingesta (`/ingest`) y el chat (`/process_input`) comparten Qdrant pero son flujos independientes

---

## Diagrama 2 — Secuencia con payloads

Cada flecha muestra el dato real que viaja en ese momento.

```mermaid
sequenceDiagram
    participant U  as 👤 Usuario
    participant FE as ⚡ Next.js
    participant API as 🚀 FastAPI
    participant RD  as 🔴 Redis
    participant SUP as Supervisor
    participant HAI as ☁️ Haiku
    participant DOC as DocExpert
    participant RET as Retriever
    participant QD  as 🟡 Qdrant
    participant SYN as Synthesizer
    participant SON as ☁️ Sonnet

    U->>FE: escribe "¿cómo detectar fallo en rodamiento?"
    Note over FE: sessionId = "b7691380-..." (en useState)

    FE->>API: POST /api/process_input
    Note over FE,API: {"query": "¿cómo detectar fallo en rodamiento?",<br/>"session_id": "b7691380-..."}

    API->>RD: GET conv:b7691380-...
    RD-->>API: lista de mensajes anteriores
    Note over RD,API: [<br/>  {"role":"user",      "content":"¿qué es el desgaste?"},<br/>  {"role":"assistant", "content":"El desgaste es..."}<br/>]

    Note over API: construye AMIAState inicial

    API->>SUP: supervisor_node(state)
    Note over API,SUP: AMIAState {<br/>  query: "¿cómo detectar fallo en rodamiento?",<br/>  conversation_history: [{...}, {...}],<br/>  retrieved_docs: [],<br/>  next_agent: "",<br/>  final_response: "",<br/>  sources: []<br/>}

    SUP->>HAI: messages.create()
    Note over SUP,HAI: model: "claude-haiku-4-5-20251001"<br/>tool_choice: {"type":"any"}<br/>tools: [route_to_agent {<br/>  next_agent: enum["doc_expert","synthesizer"],<br/>  reasoning: string<br/>}]

    HAI-->>SUP: tool_use block
    Note over HAI,SUP: {"next_agent": "doc_expert",<br/> "reasoning": "pregunta técnica sobre diagnóstico"}

    SUP-->>API: actualiza state
    Note over SUP,API: {"next_agent": "doc_expert"}

    Note over API: _route(state) → "doc_expert"

    API->>DOC: doc_expert_node(state)

    DOC->>RET: retrieve("¿cómo detectar fallo en rodamiento?", top_k=20)

    RET->>RET: SentenceTransformer.encode(query)
    Note over RET: query → vector float[384]<br/>(all-MiniLM-L6-v2, normalizado)

    RET->>QD: search(collection="documents", vector, limit=20)
    Note over RET,QD: búsqueda por similitud coseno

    QD-->>RET: 20 puntos más cercanos
    Note over QD,RET: [PointStruct {<br/>  id: "uuid",<br/>  score: 0.91,<br/>  payload: {<br/>    text: "El análisis de vibraciones...",<br/>    metadata: {source:"SKF_manual.pdf", chunk_index:42}<br/>  }<br/>}, ...]

    RET->>RET: CrossEncoder.predict([(query, chunk.text) × 20])
    Note over RET: reranking semántico profundo<br/>devuelve scores de relevancia reales

    RET-->>DOC: top 3 chunks rerankeados
    Note over RET,DOC: [Chunk {<br/>  text: "...",<br/>  metadata: {source, chunk_index},<br/>  rerank_score: 0.87<br/>}, ...]

    DOC-->>API: actualiza state
    Note over DOC,API: {"retrieved_docs": [<br/>  {"text":"...","source":"SKF_manual.pdf","rerank_score":0.87},<br/>  {"text":"...","source":"Motor_manual.pdf","rerank_score":0.74},<br/>  {"text":"...","source":"SKF_manual.pdf","rerank_score":0.61}<br/>]}

    API->>SYN: synthesizer_node(state)

    SYN->>SYN: _build_context(retrieved_docs)
    Note over SYN: "[1] Fuente: SKF_manual.pdf\nEl análisis de vibraciones...\n\n[2] Fuente: ..."

    SYN->>SON: messages.create()
    Note over SYN,SON: model: "claude-sonnet-4-6"<br/>system: "Eres experto en mantenimiento industrial..."<br/>messages: [<br/>  ...conversation_history,<br/>  {role:"user", content:"Contexto:\n[1]...[2]...\n\nPregunta: ¿cómo detectar..."}<br/>]

    SON-->>SYN: TextBlock
    Note over SON,SYN: "Los métodos principales para detectar fallos<br/>en rodamientos son [1]:\n1. Análisis de vibraciones..."

    SYN-->>API: actualiza state
    Note over SYN,API: {<br/>  "final_response": "Los métodos principales...",<br/>  "sources": [<br/>    {"index":1,"source":"SKF_manual.pdf","rerank_score":0.87},<br/>    {"index":2,"source":"Motor_manual.pdf","rerank_score":0.74}<br/>  ]<br/>}

    API->>RD: SETEX conv:b7691380-... 86400
    Note over API,RD: [...mensajes_anteriores,<br/>  {"role":"user",      "content":"¿cómo detectar fallo en rodamiento?"},<br/>  {"role":"assistant", "content":"Los métodos principales..."}<br/>]

    API-->>FE: 200 OK
    Note over API,FE: {<br/>  "response": "Los métodos principales...",<br/>  "sources": [{index, source, rerank_score}, ...],<br/>  "agent_used": "doc_expert",<br/>  "session_id": "b7691380-..."<br/>}

    FE-->>U: renderiza mensaje + fuentes colapsables
    Note over FE: <ChatMessage role="assistant" /><br/><SourceList sources={[...]} />
```

---

## Diagrama 3 — Contratos de datos

Las estructuras exactas que circulan por el sistema.

```mermaid
classDiagram

    class InputQuery {
        +str query
        +str session_id
        --
        Pydantic model
        session_id: UUID auto-generado
        si no se envía en el request
    }

    class AMIAState {
        +str query
        +list~dict~ conversation_history
        +list~dict~ retrieved_docs
        +dict|None sensor_analysis
        +dict|None economic_impact
        +dict|None work_order
        +str next_agent
        +str final_response
        +list~dict~ sources
        --
        TypedDict — contrato entre nodos
        LangGraph lo pasa a cada nodo
        cada nodo devuelve solo los campos
        que modifica (merge parcial)
    }

    class Document {
        +str content
        +dict metadata
        +str doc_id
        --
        dataclass
        doc_id = md5(bytes)
        metadata: source, type, pages
    }

    class Chunk {
        +str text
        +dict metadata
        +str chunk_id
        +list~float~ embedding
        --
        dataclass
        chunk_id = uuid5(doc_id + index)
        embedding: float[384]
        metadata hereda de Document
        + chunk_index, total_chunks
    }

    class RAGConfig {
        +int embedding_dim = 384
        +str embedding_model
        +int chunk_size = 500
        +int overlap_size = 50
        +int retrieval_top_k = 20
        +int rerank_top_k = 3
        +str llm_model
        +str qdrant_host
        +int qdrant_port
        +str collection_name
        --
        dataclass — configuración global
        se instancia una vez en main.py
        y se pasa a todos los servicios
    }

    class RedisHistory {
        +list~TurnMessage~ turns
        --
        JSON en Redis como string
        clave: conv:{session_id}
        TTL: 86400s (24h)
        máximo 10 turnos (20 mensajes)
    }

    class TurnMessage {
        +str role
        +str content
        --
        role: "user" | "assistant"
        formato nativo Anthropic API
    }

    class APIResponse {
        +str response
        +list~SourceRef~ sources
        +str agent_used
        +str session_id
        --
        JSON devuelto por POST /process_input
    }

    class SourceRef {
        +int index
        +str source
        +str page
        +float rerank_score
        --
        referencia a documento citado
        index coincide con [1],[2] en texto
    }

    Document "1" --> "N" Chunk : se divide en
    Chunk --> AMIAState : retrieved_docs
    InputQuery --> AMIAState : query + history
    RedisHistory "1" --> "N" TurnMessage : contiene
    AMIAState --> APIResponse : final_response + sources
    APIResponse "1" --> "N" SourceRef : sources
```

---

## Regla de oro para escribir código nuevo

Cuando añadas un componente nuevo (V2: sensor analyst, V3: work order agent), hazte siempre estas preguntas:

```
1. ¿Qué campos de AMIAState lee mi nodo?      → son mis INPUTS
2. ¿Qué campos de AMIAState escribe mi nodo?  → son mis OUTPUTS (dict parcial)
3. ¿Necesito un servicio externo?             → crea una clase en services/
4. ¿Cómo lo inyecto en el nodo?               → patrón factory make_X_node(servicio)
5. ¿En qué condición llega el grafo a mi nodo? → añade arista condicional en graph.py
```

Ejemplo para V2 (sensor analyst):
```python
# INPUT  → state["query"] + sensor_data (nuevo campo en AMIAState)
# OUTPUT → {"sensor_analysis": {"risk": 0.87, "failure_mode": "bearing_wear"}}
# Servicio: PredictorService (carga el modelo XGBoost)
# Factory:  make_sensor_analyst_node(predictor)
# Arista:   supervisor decide → "sensor_analyst" cuando detecta machine_id en query
```
