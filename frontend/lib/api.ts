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
  const url = new URL(`/api/v2${path}`, "http://localhost:3000");
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v != null) url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.pathname + url.search, { cache: "no-store" });
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
};
