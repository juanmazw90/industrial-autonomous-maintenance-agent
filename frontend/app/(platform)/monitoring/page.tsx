"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmt, fmtTs, fmtDuration } from "@/lib/utils";
import {
  CheckCircle2, XCircle, Clock, AlertTriangle,
  Bot, Database, FlaskConical, Activity, Cpu,
} from "lucide-react";

// ── Shared primitives ─────────────────────────────────────────────────────

function SectionTitle({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon size={14} className="text-gray-500" />
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{title}</h2>
    </div>
  );
}

function KpiCard({
  label, value, sub, accent = "text-gray-100",
}: {
  label: string; value: string; sub?: string; accent?: string;
}) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3">
      <p className="text-[11px] text-gray-600 uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold tabular-nums mt-0.5 leading-none ${accent}`}>{value}</p>
      {sub && <p className="text-[11px] text-gray-600 mt-1">{sub}</p>}
    </div>
  );
}

function StatusDot({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="w-2 h-2 rounded-full bg-gray-600 inline-block" />;
  return ok
    ? <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
    : <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse inline-block" />;
}

// ── Infrastructure section ────────────────────────────────────────────────

function InfraSection() {
  const { data, dataUpdatedAt } = useQuery({
    queryKey: ["monitoring-services"],
    queryFn: api.monitoring.services,
    refetchInterval: 30_000,
  });

  const svc = Object.fromEntries((data?.services ?? []).map((s) => [s.name, s]));

  const services = [
    { key: "postgres", label: "PostgreSQL", icon: Database },
    { key: "redis",    label: "Redis",      icon: Activity },
    { key: "qdrant",   label: "Qdrant",     icon: Database },
  ];

  return (
    <section>
      <SectionTitle icon={Activity} title="Infraestructura" />
      <div className="grid grid-cols-3 gap-3">
        {services.map(({ key, label, icon: Icon }) => {
          const s = svc[key];
          const ok = s ? s.status === "ok" : null;
          return (
            <div key={key} className={`border rounded-xl px-4 py-3 flex items-center gap-3 ${
              ok === null ? "border-gray-800 bg-gray-900/40"
              : ok ? "border-green-800/40 bg-green-950/10"
              : "border-red-800/40 bg-red-950/10"
            }`}>
              <Icon size={14} className={ok === null ? "text-gray-600" : ok ? "text-green-400" : "text-red-400"} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-300">{label}</p>
                {s?.detail && <p className="text-[11px] text-gray-600 truncate">{s.detail}</p>}
              </div>
              <StatusDot ok={ok} />
            </div>
          );
        })}
      </div>
      {dataUpdatedAt > 0 && (
        <p className="text-[11px] text-gray-700 mt-2">
          Verificado {fmtTs(new Date(dataUpdatedAt).toISOString())}
        </p>
      )}
    </section>
  );
}

// ── Plant / Fleet section ─────────────────────────────────────────────────

function PlantSection() {
  const { data } = useQuery({
    queryKey: ["operations-summary"],
    queryFn: api.operations.summary,
    refetchInterval: 30_000,
  });

  if (!data) return (
    <section>
      <SectionTitle icon={Cpu} title="Planta y Flota" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 animate-pulse">
        {[...Array(4)].map((_, i) => <div key={i} className="h-20 rounded-xl bg-gray-800/40" />)}
      </div>
    </section>
  );

  const healthAccent = data.plant_health_score >= 80 ? "text-green-400"
    : data.plant_health_score >= 60 ? "text-yellow-400" : "text-red-400";

  return (
    <section>
      <SectionTitle icon={Cpu} title="Planta y Flota" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          label="Salud de Planta"
          value={`${data.plant_health_score.toFixed(0)}%`}
          sub={`${data.machines.healthy} ok · ${data.machines.warning} aviso · ${data.machines.critical} crítico`}
          accent={healthAccent}
        />
        <KpiCard
          label="Alertas Abiertas"
          value={String(data.alerts.open)}
          sub={`${data.alerts.critical} críticas · ${data.alerts.high} altas`}
          accent={data.alerts.critical > 0 ? "text-red-400" : "text-gray-100"}
        />
        <KpiCard
          label="Órdenes Trabajo"
          value={String(data.work_orders.open)}
          sub={`de ${data.work_orders.total} totales`}
        />
        <KpiCard
          label="Exposición al Riesgo"
          value={fmt(data.risk_exposure_usd, "currency", 0)}
          accent={data.risk_exposure_usd > 50000 ? "text-yellow-400" : "text-gray-100"}
        />
      </div>
    </section>
  );
}

// ── Agents section ────────────────────────────────────────────────────────

function AgentsSection() {
  const { data } = useQuery({
    queryKey: ["agents-summary"],
    queryFn: api.agents.summary,
    refetchInterval: 30_000,
  });

  if (!data) return (
    <section>
      <SectionTitle icon={Bot} title="Agentes IA" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 animate-pulse">
        {[...Array(4)].map((_, i) => <div key={i} className="h-20 rounded-xl bg-gray-800/40" />)}
      </div>
    </section>
  );

  const totalCalls = data.totals.total_calls;
  const avgSuccess = data.totals.success_rate;
  const p95        = data.totals.latency_p95_ms;
  const slowestAgent = [...data.by_agent].sort((a, b) => (b.latency_p95_ms ?? 0) - (a.latency_p95_ms ?? 0))[0];

  return (
    <section>
      <SectionTitle icon={Bot} title="Agentes IA" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        <KpiCard label="Total Llamadas"    value={String(totalCalls)} />
        <KpiCard
          label="Tasa de Éxito"
          value={`${(avgSuccess * 100).toFixed(1)}%`}
          accent={avgSuccess >= 0.9 ? "text-green-400" : avgSuccess >= 0.7 ? "text-yellow-400" : "text-red-400"}
        />
        <KpiCard
          label="Latencia p95"
          value={fmtDuration(p95)}
          accent={p95 != null && p95 > 5000 ? "text-yellow-400" : "text-gray-100"}
        />
        <KpiCard
          label="Agente más lento"
          value={slowestAgent?.agent_name ?? "—"}
          sub={slowestAgent ? fmtDuration(slowestAgent.latency_p95_ms) : undefined}
        />
      </div>

      {/* Per-agent table */}
      <div className="border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-gray-900/80 border-b border-gray-800">
            <tr className="text-[11px] text-gray-600 uppercase tracking-wider">
              <th className="px-4 py-2.5 text-left">Agente</th>
              <th className="px-4 py-2.5 text-right">Llamadas</th>
              <th className="px-4 py-2.5 text-right">Éxito</th>
              <th className="px-4 py-2.5 text-right">p50</th>
              <th className="px-4 py-2.5 text-right">p95</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/40">
            {data.by_agent.map((a) => (
              <tr key={a.agent_name} className="hover:bg-gray-900/30">
                <td className="px-4 py-2 text-gray-400 font-mono">{a.agent_name}</td>
                <td className="px-4 py-2 text-right tabular-nums text-gray-400">{a.total_calls}</td>
                <td className={`px-4 py-2 text-right tabular-nums font-medium ${
                  a.success_rate >= 0.9 ? "text-green-400" : a.success_rate >= 0.7 ? "text-yellow-400" : "text-red-400"
                }`}>{(a.success_rate * 100).toFixed(1)}%</td>
                <td className="px-4 py-2 text-right tabular-nums text-gray-500">{fmtDuration(a.latency_p50_ms)}</td>
                <td className="px-4 py-2 text-right tabular-nums text-gray-500">{fmtDuration(a.latency_p95_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ── RAG section ───────────────────────────────────────────────────────────

function RagSection() {
  const { data } = useQuery({
    queryKey: ["rag-summary"],
    queryFn: api.rag.summary,
    refetchInterval: 30_000,
  });

  if (!data) return null;

  return (
    <section>
      <SectionTitle icon={Database} title="RAG y Caché Semántica" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="Total Consultas"  value={String(data.total_queries)} />
        <KpiCard
          label="Tasa de Aciertos"
          value={`${((1 - data.miss_rate) * 100).toFixed(1)}%`}
          sub={`${data.hits} aciertos · ${data.misses} fallos`}
          accent={(1 - data.miss_rate) >= 0.8 ? "text-green-400" : "text-yellow-400"}
        />
        <KpiCard
          label="Latencia Media"
          value={fmtDuration(data.avg_latency_ms)}
          accent={data.avg_latency_ms > 500 ? "text-yellow-400" : "text-gray-100"}
        />
        <KpiCard
          label="Similitud Media"
          value={data.avg_similarity.toFixed(3)}
          sub={data.chunk_count != null ? `${data.chunk_count.toLocaleString()} chunks` : undefined}
          accent={data.avg_similarity >= 0.7 ? "text-green-400" : "text-yellow-400"}
        />
      </div>
    </section>
  );
}

// ── ML models section ─────────────────────────────────────────────────────

function ModelsSection() {
  const { data } = useQuery({
    queryKey: ["models"],
    queryFn: api.models.list,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  if (!data) return null;

  const inProd = data.filter((m) => m.latest_stage === "Production").length;
  const inStag = data.filter((m) => m.latest_stage === "Staging").length;

  return (
    <section>
      <SectionTitle icon={FlaskConical} title="Modelos ML" />
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <KpiCard
          label="En Producción"
          value={String(inProd)}
          sub={`de ${data.length} registrados`}
          accent={inProd > 0 ? "text-green-400" : "text-yellow-400"}
        />
        <KpiCard label="En Staging" value={String(inStag)} />
        <KpiCard label="Total Modelos" value={String(data.length)} />
      </div>
      <div className="mt-3 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-gray-900/80 border-b border-gray-800">
            <tr className="text-[11px] text-gray-600 uppercase tracking-wider">
              <th className="px-4 py-2.5 text-left">Modelo</th>
              <th className="px-4 py-2.5 text-left">Versión</th>
              <th className="px-4 py-2.5 text-left">Stage</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/40">
            {data.map((m) => (
              <tr key={m.name} className="hover:bg-gray-900/30">
                <td className="px-4 py-2 text-gray-300 font-mono">{m.name}</td>
                <td className="px-4 py-2 text-gray-500">v{m.latest_version ?? "—"}</td>
                <td className="px-4 py-2">
                  <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${
                    m.latest_stage === "Production" ? "text-green-400 border-green-800/50"
                    : m.latest_stage === "Staging" ? "text-yellow-400 border-yellow-800/50"
                    : "text-gray-500 border-gray-700/50"
                  }`}>{m.latest_stage ?? "None"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function MonitoringPage() {
  const { data: services } = useQuery({
    queryKey: ["monitoring-services"],
    queryFn: api.monitoring.services,
    refetchInterval: 30_000,
  });

  const overallOk = !services || services.overall === "ok";

  return (
    <div className="px-8 py-8 max-w-5xl space-y-10">
      <div className="flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Monitorización Unificada</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Infraestructura · Planta · Agentes IA · RAG · Modelos ML
          </p>
        </div>
        <div className="flex items-center gap-2">
          {overallOk
            ? <CheckCircle2 size={16} className="text-green-400" />
            : <AlertTriangle size={16} className="text-yellow-400" />}
          <span className={`text-sm font-semibold ${overallOk ? "text-green-400" : "text-yellow-400"}`}>
            {overallOk ? "Operativo" : "Degradado"}
          </span>
        </div>
      </div>

      <InfraSection />
      <PlantSection />
      <AgentsSection />
      <RagSection />
      <ModelsSection />
    </div>
  );
}
