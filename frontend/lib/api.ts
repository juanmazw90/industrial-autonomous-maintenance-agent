// Typed API client for AMIA v2 backend endpoints.
// All functions fetch from /api/v2/* (proxied by Next.js → http://localhost:8000).

export interface OperationsSummary {
  plant_health_score: number;
  machines: { total: number; healthy: number; warning: number; critical: number };
  alerts: { open: number; new: number; critical: number; high: number; medium: number; low: number };
  work_orders: { open: number; total: number };
  risk_exposure_usd: number;
  avg_failure_probability_pct: number;
}

export interface FleetMachine {
  id: string;
  code: string;
  name: string;
  type: string;
  status: "healthy" | "warning" | "critical";
  line: { code: string; name: string };
  plant: { code: string; name: string };
  failure_probability: number | null;
  prediction_at: string | null;
  open_alerts: number;
}

export interface AssetDetail {
  id: string;
  code: string;
  name: string;
  type: string;
  status: string;
  latest_prediction: {
    id: string;
    probability: number | null;
    prediction: Record<string, unknown>;
    shap_top_features: Array<{ feature: string; value: number }> | null;
    model_name: string;
    model_version: string | null;
    created_at: string;
  } | null;
  open_alerts: Array<{ id: string; severity: string; status: string; title: string; created_at: string }>;
  open_work_orders: Array<{ id: string; title: string; priority: string; status: string; estimated_cost: number | null }>;
}

export interface TimelineEvent {
  id: string;
  ts: string;
  kind: string;
  title: string;
  payload: Record<string, unknown> | null;
  correlation_id: string | null;
}

export interface AgentsSummary {
  by_agent: Array<{
    agent_name: string;
    total_calls: number;
    success_rate: number;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
    cost_usd_total: number;
    input_tokens_total: number;
    output_tokens_total: number;
  }>;
  totals: {
    total_calls: number;
    success_rate: number;
    cost_usd_total: number;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  };
}

export interface AgentRun {
  id: string;
  session_id: string;
  agent_name: string;
  parent_run_id: string | null;
  status: "running" | "success" | "error";
  latency_ms: number | null;
  cost_usd: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  model: string | null;
  started_at: string;
  finished_at: string | null;
  error: string | null;
}

export interface ToolCall {
  id: string;
  tool_name: string;
  status: string;
  latency_ms: number | null;
  input: Record<string, unknown> | null;
  output: Record<string, unknown> | null;
  error: string | null;
  started_at: string;
}

export interface RunDetail extends AgentRun {
  input_summary: string | null;
  output_summary: string | null;
  reasoning_summary: string | null;
  retries: number;
  tool_calls: ToolCall[];
  rag_queries: Array<{
    id: string;
    query: string;
    top_k: number;
    avg_similarity: number | null;
    retrieval_latency_ms: number | null;
    hit: boolean;
    sources: string[] | null;
    created_at: string;
  }>;
}

export interface TraceNode {
  id: string;
  agent_name: string;
  status: string;
  latency_ms: number | null;
  cost_usd: number | null;
  started_at: string;
  tool_calls: Array<{ id: string; tool_name: string; status: string; latency_ms: number | null; started_at: string }>;
  children: TraceNode[];
}

export interface AlertItem {
  id: string;
  machine_code: string;
  machine_name: string;
  severity: "critical" | "high" | "medium" | "low";
  status: string;
  source: string;
  title: string;
  description: string | null;
  created_at: string;
  acknowledged_by: string | null;
  assigned_to: string | null;
  resolved_at: string | null;
}

async function get<T>(path: string, params?: Record<string, string | number | boolean | null | undefined>): Promise<T> {
  const search = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v != null) search.set(k, String(v));
    });
  }
  const qs = search.toString();
  const prefix = path.startsWith("/api/") ? "" : "/api/v2";
  const res = await fetch(`${prefix}${path}${qs ? `?${qs}` : ""}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api/v2${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API POST ${path} → ${res.status}`);
  return res.json();
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api/v2${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API PATCH ${path} → ${res.status}`);
  return res.json();
}

// ── ML / XAI / RAG / Evaluation / Logs types ─────────────────────────────────

export interface MLModel {
  name: string;
  description: string | null;
  latest_version: string | null;
  latest_stage: string | null;
  latest_run_id: string | null;
  tags: Record<string, string>;
}

export interface ModelMetrics {
  model: string;
  version: string;
  run_id: string;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
}

export interface DriftSummary {
  model: string;
  status: string;
  message?: string;
  dataset_drift?: boolean;
  share_drifted_features?: number;
  number_of_columns?: number;
  number_of_drifted_columns?: number;
  feature_drift?: Record<string, { drift_detected: boolean; p_value: number; stattest: string }>;
  generated_at?: string;
}

export interface PredictionExplanation {
  prediction_id: string;
  machine_id: string;
  model_name: string;
  probability: number | null;
  prediction: Record<string, unknown>;
  shap_top_features: Array<{ feature: string; value: number }>;
  created_at: string;
  similar_predictions: Array<{
    id: string;
    probability: number | null;
    shap_top_features: Array<{ feature: string; value: number }>;
    created_at: string;
  }>;
}

export interface RagSummary {
  total_queries: number;
  avg_latency_ms: number;
  avg_similarity: number;
  miss_rate: number;
  hits: number;
  misses: number;
  chunk_count: number | null;
}

export interface RagQuery {
  id: string;
  agent_run_id: string | null;
  query: string;
  top_k: number;
  retrieval_latency_ms: number | null;
  avg_similarity: number | null;
  hit: boolean;
  sources: string[] | null;
  created_at: string;
}

export interface Evaluation {
  id: string;
  agent_run_id: string;
  accuracy: number | null;
  groundedness: number | null;
  hallucination_flag: boolean;
  tool_success_rate: number | null;
  evaluator: string | null;
  notes: string | null;
  created_at: string;
}

export interface LogEntry {
  timestamp?: string;
  level?: string;
  event?: string;
  message?: string;
  logger?: string;
  [key: string]: unknown;
}

export interface AuditEntry {
  id: string;
  action: string;
  entity_type: string;
  entity_id: string | null;
  actor_id: string | null;
  actor_type: string;
  diff: Record<string, unknown> | null;
  correlation_id: string | null;
  created_at: string;
}

export interface RULPrediction {
  machine_id: string;
  hours_remaining: number;
  degradation_fraction: number;
  urgency_level: "critical" | "warning" | "normal";
  confidence: number;
  as_of_timestamp: string;
}

export interface GlobalTimelineEvent {
  id: string;
  ts: string;
  kind: string;
  title: string;
  payload: Record<string, unknown> | null;
  correlation_id: string | null;
  machine_code: string;
  machine_name: string;
}

export type WorkOrderStatus = "open" | "assigned" | "in_progress" | "completed";
export type WorkOrderPriority = "critical" | "high" | "medium" | "low";

export interface WorkOrder {
  id: string;
  machine_code: string;
  machine_name: string;
  title: string;
  description: string | null;
  priority: WorkOrderPriority;
  status: WorkOrderStatus;
  assigned_to: string | null;
  estimated_cost: number | null;
  alert_id: string | null;
  created_at: string;
  updated_at: string | null;
}

// ── Operations ────────────────────────────────────────────────────────────────
export const api = {
  operations: {
    summary: () => get<OperationsSummary>("/operations/summary"),
  },
  fleet: {
    list: (status?: string) => get<FleetMachine[]>("/fleet", { status }),
  },
  assets: {
    get: (code: string) => get<AssetDetail>(`/assets/${code}`),
    timeline: (code: string, kind?: string, limit = 50) =>
      get<{ total: number; events: TimelineEvent[] }>(`/assets/${code}/timeline`, { kind, limit }),
  },
  agents: {
    summary: () => get<AgentsSummary>("/agents/summary"),
    runs: (params?: { agent?: string; status?: string; limit?: number; offset?: number }) =>
      get<{ total: number; runs: AgentRun[] }>("/agents/runs", params as Record<string, string | number>),
    run: (id: string) => get<RunDetail>(`/agents/runs/${id}`),
    trace: (id: string) => get<{ session_id: string; trace: TraceNode }>(`/agents/runs/${id}/trace`),
  },
  alerts: {
    list: (params?: { status?: string; severity?: string; limit?: number; offset?: number }) =>
      get<{ total: number; alerts: AlertItem[] }>("/alerts", params as Record<string, string | number>),
    transition: (id: string, status: string) => patch<AlertItem>(`/alerts/${id}`, { status }),
  },
  monitoring: {
    services: () =>
      get<{ overall: string; services: Array<{ name: string; status: string; detail?: string }>; checked_at: string }>(
        "/monitoring/services"
      ),
  },
  models: {
    list: () => get<MLModel[]>("/models"),
    metrics: (name: string) => get<ModelMetrics>(`/models/${encodeURIComponent(name)}/metrics`),
    drift: (name: string) => get<DriftSummary>(`/models/${encodeURIComponent(name)}/drift`),
  },
  predictions: {
    explain: (id: string) => get<PredictionExplanation>(`/predictions/${id}/explain`),
  },
  rag: {
    summary: () => get<RagSummary>("/rag/summary"),
    queries: (params?: { limit?: number; offset?: number; hit?: boolean }) =>
      get<{ total: number; queries: RagQuery[] }>("/rag/queries", params as Record<string, string | number | boolean>),
    documents: (params?: { limit?: number; offset?: number; search?: string }) =>
      get<{ limit: number; offset: number; documents: Array<{ id: string; payload: Record<string, unknown> }> }>(
        "/rag/documents",
        params as Record<string, string | number>
      ),
  },
  evaluations: {
    list: (params?: { agent_run_id?: string; limit?: number; offset?: number }) =>
      get<{ total: number; evaluations: Evaluation[] }>("/agents/evaluations", params as Record<string, string | number>),
    create: (body: {
      agent_run_id: string;
      accuracy?: number;
      groundedness?: number;
      hallucination_flag?: boolean;
      tool_success_rate?: number;
      evaluator?: string;
      notes?: string;
    }) => post<{ id: string; agent_run_id: string; created_at: string }>("/agents/evaluations", body),
  },
  logs: {
    list: (params?: { limit?: number; level?: string }) =>
      get<{ source: string; count: number; entries: LogEntry[] }>("/logs", params as Record<string, string | number>),
  },
  audit: {
    list: (params?: { entity_type?: string; actor_id?: string; limit?: number; offset?: number }) =>
      get<{ total: number; entries: AuditEntry[] }>("/audit", params as Record<string, string | number>),
  },
  rul: {
    // Endpoint legacy v1: pasa por el rewrite /api/:path* (sin prefijo /api/v2)
    all: (): Promise<RULPrediction[]> => get<RULPrediction[]>("/api/predict/rul/all"),
  },
  timeline: {
    list: (params?: { kind?: string; machine_code?: string; limit?: number; offset?: number }) =>
      get<{ total: number; limit: number; offset: number; events: GlobalTimelineEvent[] }>(
        "/timeline",
        params as Record<string, string | number>
      ),
  },
  workOrders: {
    list: (params?: { status?: string; priority?: string; machine_code?: string; limit?: number; offset?: number }) =>
      get<{ total: number; work_orders: WorkOrder[] }>("/work-orders", params as Record<string, string | number>),
    create: (body: { machine_code: string; title: string; description?: string; priority?: string; estimated_cost?: number }) =>
      post<WorkOrder>("/work-orders", body),
    update: (id: string, body: { status?: string; assigned_to?: string; estimated_cost?: number }) =>
      patch<WorkOrder>(`/work-orders/${id}`, body),
  },
};
