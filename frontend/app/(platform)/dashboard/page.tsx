"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type OperationsSummary } from "@/lib/api";
import { fmt, fmtTs } from "@/lib/utils";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  Wrench,
  DollarSign,
} from "lucide-react";

// ── KPI tile ──────────────────────────────────────────────────────────────────

function Tile({
  label,
  value,
  sub,
  color = "text-gray-100",
  icon: Icon,
  pulse,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  icon?: React.ElementType;
  pulse?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5 bg-gray-900/60 border border-gray-800 rounded-xl px-5 py-4 min-w-0">
      <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wider">
        {Icon && <Icon size={12} className="shrink-0" />}
        {label}
      </div>
      <p className={`text-3xl font-bold tabular-nums leading-none ${color} ${pulse ? "animate-pulse" : ""}`}>
        {value}
      </p>
      {sub && <p className="text-xs text-gray-600">{sub}</p>}
    </div>
  );
}

// ── Alert severity row ────────────────────────────────────────────────────────

function AlertBar({ alerts }: { alerts: OperationsSummary["alerts"] }) {
  const bars = [
    { label: "Crítico", count: alerts.critical, color: "bg-red-500" },
    { label: "Alto",    count: alerts.high,     color: "bg-orange-500" },
    { label: "Medio",   count: alerts.medium,   color: "bg-yellow-500" },
    { label: "Bajo",    count: alerts.low,       color: "bg-blue-500" },
  ];
  return (
    <div className="flex gap-2 items-center flex-wrap">
      {bars.map(({ label, count, color }) => (
        <div key={label} className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${color}`} />
          <span className="text-xs text-gray-400">{label}: <span className="font-semibold text-gray-200">{count}</span></span>
        </div>
      ))}
    </div>
  );
}

// ── Machine status visual ─────────────────────────────────────────────────────

function MachineStatusBar({ machines }: { machines: OperationsSummary["machines"] }) {
  const { total, healthy, warning, critical } = machines;
  if (!total) return null;
  return (
    <div className="space-y-3">
      <div className="flex gap-1 h-3 rounded-full overflow-hidden">
        <div className="bg-green-500 transition-all" style={{ width: `${(healthy / total) * 100}%` }} />
        <div className="bg-yellow-500 transition-all" style={{ width: `${(warning / total) * 100}%` }} />
        <div className="bg-red-500 transition-all" style={{ width: `${(critical / total) * 100}%` }} />
      </div>
      <div className="flex gap-4 text-xs">
        <span className="text-green-400">● {healthy} saludable</span>
        <span className="text-yellow-400">● {warning} advertencia</span>
        <span className="text-red-400">● {critical} crítico</span>
      </div>
    </div>
  );
}

// ── Health score gauge (simple ring) ─────────────────────────────────────────

function HealthScore({ score }: { score: number }) {
  const color = score >= 80 ? "#22c55e" : score >= 50 ? "#eab308" : "#ef4444";
  const r = 40;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-28 h-28">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r={r} fill="none" stroke="#1f2937" strokeWidth="12" />
          <circle
            cx="50" cy="50" r={r} fill="none"
            stroke={color} strokeWidth="12"
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold tabular-nums" style={{ color }}>{score.toFixed(0)}</span>
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">%</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 uppercase tracking-wider">Salud de Planta</p>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 animate-pulse">
      {[...Array(8)].map((_, i) => (
        <div key={i} className="h-24 rounded-xl bg-gray-800/50" />
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const { data, isLoading, error, dataUpdatedAt } = useQuery({
    queryKey: ["operations-summary"],
    queryFn: api.operations.summary,
    refetchInterval: 15_000,
  });

  return (
    <div className="px-8 py-8 max-w-6xl">
      {/* Page header */}
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Centro de Operaciones</h1>
          <p className="text-sm text-gray-500 mt-0.5">Estado de planta y exposición al riesgo en tiempo real</p>
        </div>
        {dataUpdatedAt > 0 && (
          <p className="text-xs text-gray-600">Actualizado {fmtTs(new Date(dataUpdatedAt).toISOString(), { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</p>
        )}
      </div>

      {isLoading && <Skeleton />}
      {error && (
        <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400 mb-6">
          No se puede conectar al backend — asegúrese de que el servidor está corriendo en el puerto 8000.
        </div>
      )}

      {data && (
        <>
          {/* Row 1: Health score + KPI grid */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
            {/* Health gauge */}
            <div className="flex items-center justify-center lg:col-span-1 bg-gray-900/60 border border-gray-800 rounded-xl p-6">
              <HealthScore score={data.plant_health_score} />
            </div>

            {/* KPI tiles */}
            <div className="lg:col-span-4 grid grid-cols-2 md:grid-cols-4 gap-4">
              <Tile
                label="Total de Máquinas"
                value={String(data.machines.total)}
                sub={`${data.machines.healthy} saludable`}
                icon={Activity}
              />
              <Tile
                label="Alertas Abiertas"
                value={String(data.alerts.open)}
                sub={`${data.alerts.new} nuevas`}
                color={data.alerts.critical > 0 ? "text-red-400" : data.alerts.open > 0 ? "text-yellow-400" : "text-green-400"}
                icon={AlertTriangle}
                pulse={data.alerts.critical > 0}
              />
              <Tile
                label="Órdenes de Trabajo"
                value={String(data.work_orders.open)}
                sub={`de ${data.work_orders.total} en total`}
                icon={Wrench}
              />
              <Tile
                label="Exposición al Riesgo"
                value={fmt(data.risk_exposure_usd, "currency")}
                sub="costo de OTs abiertas"
                color={data.risk_exposure_usd > 5000 ? "text-orange-400" : "text-gray-100"}
                icon={DollarSign}
              />
            </div>
          </div>

          {/* Row 2: machine status + alert breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-2">
                <Cpu size={12} /> Estado de Flota
              </h2>
              <MachineStatusBar machines={data.machines} />
              <p className="text-xs text-gray-600">
                Prob. media de fallo: <span className="text-gray-300 font-semibold">{data.avg_failure_probability_pct}%</span>
              </p>
            </div>

            <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 space-y-4">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-2">
                <AlertTriangle size={12} /> Desglose de Alertas
              </h2>
              <AlertBar alerts={data.alerts} />
              {data.alerts.open === 0 && (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <CheckCircle2 size={14} />
                  Todo en orden — sin alertas abiertas
                </div>
              )}
            </div>
          </div>

          {/* Critical machines warning */}
          {data.machines.critical > 0 && (
            <div className="rounded-xl border border-red-700/60 bg-red-950/30 px-5 py-4 flex items-start gap-3">
              <AlertTriangle size={16} className="text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-red-400">
                  {data.machines.critical} máquina{data.machines.critical > 1 ? "s" : ""} en estado crítico
                </p>
                <p className="text-xs text-red-300/60 mt-0.5">
                  Navegue a Flota para identificar los activos afectados y tomar acción.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Cpu({ size, className }: { size: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className={className}>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2" />
    </svg>
  );
}
