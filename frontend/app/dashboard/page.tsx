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

// ── API ────────────────────────────────────────────────────────────────────

async function fetchPredictions(): Promise<MachinePrediction[]> {
  const res = await fetch("/api/predict/failure/all", { cache: "no-store" });
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

// ── Page ───────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [predictions, setPredictions] = useState<MachinePrediction[]>([]);
  const [lastUpdate, setLastUpdate]   = useState<Date | null>(null);
  const [error, setError]             = useState<string | null>(null);

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

  const criticalMachines = predictions.filter((p) => p.alert_level === "red");

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
