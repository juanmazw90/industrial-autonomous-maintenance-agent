"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────

interface MachinePrediction {
  machine_id: string;
  failure_probability: number;
  risk_score: number;
  alert_level: "green" | "yellow" | "red";
  is_high_risk: boolean;
  threshold_used: number;
  as_of_timestamp: string;
}

interface WorkOrder {
  work_order_id: string;
  machine_id: string;
  failure_mode: string;
  priority: "urgent" | "high" | "normal";
  estimated_cost: number;
  sop_reference: string;
  status: "open" | "completed";
  created_at: string;
}

interface RULPrediction {
  machine_id: string;
  hours_remaining: number;
  degradation_fraction: number;
  urgency_level: "critical" | "warning" | "normal";
  as_of_timestamp: string;
}

// ── Config ─────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 3000;

const MACHINE_TYPE: Record<string, string> = {
  "COMP-001":  "Compresor",
  "MOTOR-001": "Motor de Inducción",
  "MOTOR-002": "Motor de Inducción",
  "PUMP-001":  "Bomba Centrífuga",
  "PUMP-002":  "Bomba Centrífuga",
};

const ALERT_CONFIG = {
  green: {
    label:      "NORMAL",
    dot:        "bg-green-400",
    card:       "bg-green-950/40 border-green-800/60",
    text:       "text-green-400",
    bar:        "bg-green-500",
    pulse:      "",
  },
  yellow: {
    label:      "ATENCIÓN",
    dot:        "bg-yellow-400",
    card:       "bg-yellow-950/40 border-yellow-700/60",
    text:       "text-yellow-400",
    bar:        "bg-yellow-500",
    pulse:      "animate-pulse",
  },
  red: {
    label:      "CRÍTICO",
    dot:        "bg-red-500",
    card:       "bg-red-950/40 border-red-700/60",
    text:       "text-red-400",
    bar:        "bg-red-500",
    pulse:      "animate-pulse",
  },
};

const PRIORITY_CONFIG = {
  urgent: { label: "URGENTE", badge: "bg-red-950/60 text-red-400 border-red-700/60" },
  high:   { label: "ALTA",    badge: "bg-yellow-950/60 text-yellow-400 border-yellow-700/60" },
  normal: { label: "NORMAL",  badge: "bg-green-950/60 text-green-400 border-green-800/60" },
};

const WO_POLL_INTERVAL_MS  = 5000;
const RUL_POLL_INTERVAL_MS = 5000;
const MAX_RUL_HOURS        = 500;

const URGENCY_CONFIG = {
  critical: {
    label: "CRÍTICO",
    text:  "text-red-400",
    bar:   "bg-red-500",
    card:  "bg-red-950/40 border-red-700/60",
    pulse: "animate-pulse",
  },
  warning: {
    label: "MODERADO",
    text:  "text-yellow-400",
    bar:   "bg-yellow-500",
    card:  "bg-yellow-950/40 border-yellow-700/60",
    pulse: "animate-pulse",
  },
  normal: {
    label: "NORMAL",
    text:  "text-green-400",
    bar:   "bg-green-500",
    card:  "bg-gray-900/40 border-gray-700/60",
    pulse: "",
  },
};

// ── API ────────────────────────────────────────────────────────────────────

async function fetchPredictions(): Promise<MachinePrediction[]> {
  const res = await fetch("/api/predict/failure/all", { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchWorkOrders(): Promise<WorkOrder[]> {
  const res = await fetch("/api/work-orders", { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function completeWorkOrder(woId: string): Promise<void> {
  const res = await fetch(`/api/work-orders/${woId}/complete`, { method: "PATCH" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}

async function fetchRULPredictions(): Promise<RULPrediction[]> {
  const res = await fetch("/api/predict/rul/all", { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Components ─────────────────────────────────────────────────────────────

function RiskBar({ probability }: { probability: number }) {
  const pct = Math.round(probability * 100);
  const color =
    pct >= 70 ? "bg-red-500" : pct >= 35 ? "bg-yellow-500" : "bg-green-500";
  return (
    <div className="w-full bg-gray-800 rounded-full h-1.5 mt-1">
      <div
        className={`h-1.5 rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  );
}

function MachineCard({ machine }: { machine: MachinePrediction }) {
  const cfg   = ALERT_CONFIG[machine.alert_level];
  const pct   = Math.round(machine.failure_probability * 100);
  const mtype = MACHINE_TYPE[machine.machine_id] ?? "Equipo Industrial";

  // Formatear timestamp — solo hora si es hoy
  const ts = machine.as_of_timestamp
    ? new Date(machine.as_of_timestamp).toLocaleString("es-ES", {
        month: "short",
        day:   "numeric",
        hour:  "2-digit",
        minute:"2-digit",
      })
    : "—";

  return (
    <div
      className={`border rounded-xl p-5 flex flex-col gap-3 transition-colors duration-500 ${cfg.card}`}
    >
      {/* Header: ID + indicador */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-lg font-bold text-gray-100 leading-none">
            {machine.machine_id}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">{mtype}</p>
        </div>
        <span
          className={`flex items-center gap-1.5 text-xs font-semibold px-2 py-1 rounded-full border ${cfg.text} border-current/30`}
        >
          <span className={`w-2 h-2 rounded-full ${cfg.dot} ${cfg.pulse}`} />
          {cfg.label}
        </span>
      </div>

      {/* Probabilidad */}
      <div>
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-gray-500">Prob. fallo 24h</span>
          <span className={`text-2xl font-bold tabular-nums ${cfg.text}`}>
            {pct}%
          </span>
        </div>
        <RiskBar probability={machine.failure_probability} />
      </div>

      {/* Umbral */}
      <div className="flex justify-between text-xs text-gray-600">
        <span>Umbral de alerta: {Math.round(machine.threshold_used * 100)}%</span>
        <span>{ts}</span>
      </div>
    </div>
  );
}

function StatusBar({
  predictions,
  lastUpdate,
  error,
}: {
  predictions: MachinePrediction[];
  lastUpdate: Date | null;
  error: string | null;
}) {
  const counts = predictions.reduce(
    (acc, p) => { acc[p.alert_level]++; return acc; },
    { green: 0, yellow: 0, red: 0 } as Record<string, number>
  );

  return (
    <div className="flex items-center gap-4 text-xs text-gray-500">
      <span className="text-green-400 font-medium">🟢 {counts.green}</span>
      <span className="text-yellow-400 font-medium">🟡 {counts.yellow}</span>
      <span className="text-red-400 font-medium">🔴 {counts.red}</span>
      <span className="border-l border-gray-700 pl-4">
        {error
          ? <span className="text-red-400">Sin conexión con backend</span>
          : lastUpdate
          ? `Actualizado ${lastUpdate.toLocaleTimeString("es-ES")}`
          : "Cargando…"
        }
      </span>
    </div>
  );
}

function WorkOrderCard({
  order,
  onComplete,
}: {
  order: WorkOrder;
  onComplete: (id: string) => void;
}) {
  const cfg = PRIORITY_CONFIG[order.priority] ?? PRIORITY_CONFIG.normal;
  const ts = new Date(order.created_at).toLocaleString("es-ES", {
    month:  "short",
    day:    "numeric",
    hour:   "2-digit",
    minute: "2-digit",
  });
  const cost = order.estimated_cost.toLocaleString("es-ES", {
    style:              "currency",
    currency:           "USD",
    maximumFractionDigits: 0,
  });

  return (
    <div className="border border-gray-700/60 rounded-xl p-4 flex flex-col gap-3 bg-gray-900/40">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-bold text-gray-100 leading-none">
            {order.work_order_id}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">{order.machine_id}</p>
        </div>
        <span className={`text-xs font-semibold px-2 py-1 rounded-full border ${cfg.badge}`}>
          {cfg.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <span className="text-gray-500">Modo de fallo</span>
        <span className="text-gray-300 font-medium">{order.failure_mode.replace(/_/g, " ")}</span>
        <span className="text-gray-500">SOP</span>
        <span className="text-gray-300 font-mono">{order.sop_reference}</span>
        <span className="text-gray-500">Costo estimado</span>
        <span className="text-yellow-400 font-semibold tabular-nums">{cost}</span>
        <span className="text-gray-500">Creada</span>
        <span className="text-gray-400">{ts}</span>
      </div>

      <button
        onClick={() => onComplete(order.work_order_id)}
        className="w-full text-xs font-medium py-1.5 rounded-lg border border-gray-700 text-gray-400 hover:text-gray-100 hover:border-gray-500 hover:bg-gray-800 transition-colors"
      >
        Marcar completada
      </button>
    </div>
  );
}

function RULBar({ fraction }: { fraction: number }) {
  const pct   = Math.round(Math.min(fraction, 1) * 100);
  const color = pct >= 80 ? "bg-red-500" : pct >= 40 ? "bg-yellow-500" : "bg-green-500";
  return (
    <div className="w-full bg-gray-800 rounded-full h-1.5 mt-1">
      <div
        className={`h-1.5 rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function RULCard({ rul }: { rul: RULPrediction }) {
  const cfg       = URGENCY_CONFIG[rul.urgency_level] ?? URGENCY_CONFIG.normal;
  const mtype     = MACHINE_TYPE[rul.machine_id] ?? "Equipo Industrial";
  const degradPct = Math.round(rul.degradation_fraction * 100);

  return (
    <div className={`border rounded-xl p-5 flex flex-col gap-3 transition-colors duration-500 ${cfg.card}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-lg font-bold text-gray-100 leading-none">{rul.machine_id}</p>
          <p className="text-xs text-gray-500 mt-0.5">{mtype}</p>
        </div>
        <span className={`flex items-center gap-1.5 text-xs font-semibold px-2 py-1 rounded-full border ${cfg.text} border-current/30 ${cfg.pulse}`}>
          {cfg.label}
        </span>
      </div>

      <div>
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-gray-500">Vida útil restante</span>
          <span className={`text-2xl font-bold tabular-nums ${cfg.text}`}>
            {rul.hours_remaining.toFixed(0)}h
          </span>
        </div>
        <RULBar fraction={rul.degradation_fraction} />
      </div>

      <div className="flex justify-between text-xs text-gray-600">
        <span>Degradación acumulada: {degradPct}%</span>
        <span>de {MAX_RUL_HOURS}h máx.</span>
      </div>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [predictions, setPredictions]   = useState<MachinePrediction[]>([]);
  const [lastUpdate, setLastUpdate]     = useState<Date | null>(null);
  const [error, setError]               = useState<string | null>(null);
  const [workOrders, setWorkOrders]     = useState<WorkOrder[]>([]);
  const [rulPredictions, setRulPredictions] = useState<RULPrediction[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await fetchPredictions();
        if (!cancelled) {
          setPredictions(data);
          setLastUpdate(new Date());
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Error");
      }
    }

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function pollWO() {
      try {
        const data = await fetchWorkOrders();
        if (!cancelled) setWorkOrders(data);
      } catch { /* fallo silencioso: el panel de WO es secundario */ }
    }

    pollWO();
    const id = setInterval(pollWO, WO_POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function pollRUL() {
      try {
        const data = await fetchRULPredictions();
        if (!cancelled) setRulPredictions(data);
      } catch { /* fallo silencioso: RUL puede no estar entrenado */ }
    }

    pollRUL();
    const id = setInterval(pollRUL, RUL_POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  async function handleComplete(woId: string) {
    try {
      await completeWorkOrder(woId);
      setWorkOrders((prev) =>
        prev.map((o) => o.work_order_id === woId ? { ...o, status: "completed" } : o)
      );
    } catch { /* fallo silencioso */ }
  }

  const criticalMachines = predictions.filter((p) => p.alert_level === "red");
  const openOrders = workOrders.filter((o) => o.status === "open");
  const totalRisk  = openOrders.reduce((sum, o) => sum + o.estimated_cost, 0);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="flex-none border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400" />
          <span className="text-gray-400 text-sm">Dashboard de Monitoreo</span>
        </div>
        <StatusBar predictions={predictions} lastUpdate={lastUpdate} error={error} />
      </header>

      <main className="flex-1 px-6 py-8 max-w-6xl mx-auto w-full">

        {/* Alertas críticas */}
        {criticalMachines.length > 0 && (
          <div className="mb-6 rounded-xl bg-red-950/60 border border-red-700/60 px-5 py-4">
            <p className="text-red-400 font-semibold text-sm mb-1">
              ⚠️ {criticalMachines.length} máquina{criticalMachines.length > 1 ? "s" : ""} en estado crítico
            </p>
            <p className="text-red-300/70 text-xs">
              {criticalMachines.map((m) => m.machine_id).join(", ")} —{" "}
              probabilidad de fallo superior al umbral de alerta.
              Revisar inmediatamente.
            </p>
          </div>
        )}

        {/* Grid de tarjetas */}
        {predictions.length === 0 && !error ? (
          <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
            Conectando con el predictor…
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3 text-gray-500 text-sm">
            <p className="text-red-400">{error}</p>
            <p>Asegúrate de que el backend está corriendo en localhost:8000</p>
            <code className="text-xs bg-gray-800 px-3 py-1.5 rounded text-gray-400">
              uv run uvicorn app.main:app --app-dir backend --port 8000
            </code>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {predictions.map((m) => (
              <MachineCard key={m.machine_id} machine={m} />
            ))}
          </div>
        )}

        {/* Sección: Órdenes de Trabajo */}
        {(predictions.length > 0 || workOrders.length > 0) && (
          <section className="mt-10">
            <div className="flex items-baseline justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                Órdenes de Trabajo Activas
              </h2>
              {openOrders.length > 0 && (
                <span className="text-xs text-gray-500">
                  Impacto económico en riesgo:{" "}
                  <span className="text-yellow-400 font-bold tabular-nums">
                    {totalRisk.toLocaleString("es-ES", {
                      style: "currency",
                      currency: "USD",
                      maximumFractionDigits: 0,
                    })}
                  </span>
                </span>
              )}
            </div>

            {openOrders.length === 0 ? (
              <div className="rounded-xl border border-gray-800 bg-gray-900/20 py-10 text-center text-sm text-gray-600">
                Sin alertas activas — todas las máquinas operando dentro de parámetros normales
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {openOrders.map((order) => (
                  <WorkOrderCard
                    key={order.work_order_id}
                    order={order}
                    onComplete={handleComplete}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {/* Sección: Vida Útil Restante */}
        {rulPredictions.length > 0 && (
          <section className="mt-10">
            <div className="flex items-baseline justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                Vida Útil Restante (RUL)
              </h2>
              <span className="text-xs text-gray-500">
                {rulPredictions.filter((r) => r.urgency_level !== "normal").length > 0
                  ? `${rulPredictions.filter((r) => r.urgency_level !== "normal").length} máquina(s) con degradación activa`
                  : "Todas las máquinas dentro de parámetros"}
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {rulPredictions.map((r) => (
                <RULCard key={r.machine_id} rul={r} />
              ))}
            </div>
          </section>
        )}

        {/* Leyenda */}
        {predictions.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-6 text-xs text-gray-600">
            <span>
              <span className="text-green-400 font-medium">🟢 Normal</span>
              {" "}— prob. &lt; 35%
            </span>
            <span>
              <span className="text-yellow-400 font-medium">🟡 Atención</span>
              {" "}— prob. 35–70%
            </span>
            <span>
              <span className="text-red-400 font-medium">🔴 Crítico</span>
              {" "}— prob. &gt; 70%
            </span>
            <span className="border-l border-gray-700 pl-6">
              Actualización automática cada {POLL_INTERVAL_MS / 1000}s
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
