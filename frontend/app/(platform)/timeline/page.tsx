"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, type GlobalTimelineEvent } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { Activity, Filter, Search } from "lucide-react";

// ── Event kind config ─────────────────────────────────────────────────────

const KIND_CFG: Record<string, { label: string; dot: string; border: string }> = {
  sensor_anomaly: { label: "Anomalía sensor",    dot: "bg-orange-400",  border: "border-orange-800/40" },
  prediction:     { label: "Predicción",          dot: "bg-indigo-400",  border: "border-indigo-800/40" },
  agent_decision: { label: "Decisión agente",     dot: "bg-blue-400",    border: "border-blue-800/40"   },
  rca:            { label: "Análisis RCA",        dot: "bg-purple-400",  border: "border-purple-800/40" },
  economic:       { label: "Impacto económico",   dot: "bg-yellow-400",  border: "border-yellow-800/40" },
  wo_created:     { label: "OT creada",           dot: "bg-green-400",   border: "border-green-800/40"  },
  alert_created:  { label: "Alerta creada",       dot: "bg-red-400",     border: "border-red-800/40"    },
  alert_resolved: { label: "Alerta resuelta",     dot: "bg-teal-400",    border: "border-teal-800/40"   },
};

const ALL_KINDS = Object.keys(KIND_CFG);

function KindBadge({ kind }: { kind: string }) {
  const cfg = KIND_CFG[kind] ?? { label: kind, dot: "bg-gray-400", border: "border-gray-700" };
  return (
    <span className={`flex items-center gap-1.5 text-[11px] font-medium text-gray-400 border rounded-full px-2 py-0.5 ${cfg.border} bg-gray-900/60`}>
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
}

function TimelineItem({ event }: { event: GlobalTimelineEvent }) {
  const cfg = KIND_CFG[event.kind] ?? { dot: "bg-gray-500", label: event.kind, border: "border-gray-700" };
  return (
    <div className="flex gap-4 group">
      {/* Timeline spine */}
      <div className="flex flex-col items-center shrink-0 w-5">
        <div className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${cfg.dot}`} />
        <div className="w-px flex-1 bg-gray-800 mt-1" />
      </div>

      {/* Content */}
      <div className="pb-5 flex-1 min-w-0">
        <div className="flex items-start justify-between gap-3 flex-wrap mb-1.5">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-gray-200">{event.title}</span>
            <KindBadge kind={event.kind} />
          </div>
          <time className="text-[11px] text-gray-600 whitespace-nowrap shrink-0">{fmtTs(event.ts)}</time>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-gray-600 bg-gray-800/60 px-1.5 py-0.5 rounded">
            {event.machine_code}
          </span>
          <span className="text-[11px] text-gray-700">{event.machine_name}</span>
          {event.correlation_id && (
            <span className="text-[11px] text-gray-700 font-mono">
              · corr {event.correlation_id.slice(0, 8)}
            </span>
          )}
        </div>

        {event.payload && Object.keys(event.payload).length > 0 && (
          <details className="mt-2">
            <summary className="text-[11px] text-gray-600 cursor-pointer hover:text-gray-400 select-none">
              Ver detalles
            </summary>
            <pre className="mt-1.5 text-[10px] text-gray-600 bg-gray-900/60 rounded-lg p-2 overflow-x-auto max-h-32 font-mono">
              {JSON.stringify(event.payload, null, 2)}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

export default function TimelinePage() {
  const [kindFilter, setKindFilter]       = useState("");
  const [machineFilter, setMachineFilter] = useState("");
  const [search, setSearch]               = useState("");
  const [page, setPage]                   = useState(0);
  const PAGE_SIZE = 50;

  const { data, isLoading } = useQuery({
    queryKey: ["global-timeline", kindFilter, machineFilter, page],
    queryFn: () =>
      api.timeline.list({
        kind:         kindFilter   || undefined,
        machine_code: machineFilter || undefined,
        limit:        PAGE_SIZE,
        offset:       page * PAGE_SIZE,
      }),
    refetchInterval: 20_000,
  });

  const events = (data?.events ?? []).filter(
    (e) => !search || e.title.toLowerCase().includes(search.toLowerCase()) || e.machine_code.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="px-8 py-8 max-w-3xl">
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Línea de Tiempo Global</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Todos los eventos de la planta — {data?.total ?? "…"} en total
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-gray-600">En vivo</span>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-8 flex-wrap">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm flex-1 min-w-48">
          <Search size={13} className="text-gray-500 shrink-0" />
          <input
            type="text"
            placeholder="Buscar eventos o máquinas…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent outline-none text-gray-200 placeholder-gray-600 w-full text-sm"
          />
        </div>
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <Filter size={12} className="text-gray-500" />
          <select
            value={kindFilter}
            onChange={(e) => { setKindFilter(e.target.value); setPage(0); }}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">Todos los tipos</option>
            {ALL_KINDS.map((k) => (
              <option key={k} value={k}>{KIND_CFG[k].label}</option>
            ))}
          </select>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <select
            value={machineFilter}
            onChange={(e) => { setMachineFilter(e.target.value); setPage(0); }}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">Todas las máquinas</option>
            {["COMP-001", "MOTOR-001", "MOTOR-002", "PUMP-001", "PUMP-002"].map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Timeline */}
      {isLoading && (
        <div className="space-y-6">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex gap-4 animate-pulse">
              <div className="w-2.5 h-2.5 rounded-full bg-gray-700 mt-1.5 shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-800 rounded w-3/4" />
                <div className="h-2 bg-gray-800/60 rounded w-1/2" />
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && events.length === 0 && (
        <div className="py-16 flex flex-col items-center gap-3 text-center">
          <Activity size={28} className="text-gray-700" />
          <p className="text-sm text-gray-500">Sin eventos que mostrar.</p>
          <p className="text-xs text-gray-700">Los eventos aparecen cuando el agente IA procesa alertas o el motor de alertas detecta anomalías.</p>
        </div>
      )}

      {!isLoading && events.length > 0 && (
        <div>
          {events.map((e) => (
            <TimelineItem key={e.id} event={e} />
          ))}

          {/* Pagination */}
          {data && data.total > PAGE_SIZE && (
            <div className="flex items-center justify-between pt-4 mt-2 border-t border-gray-800">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30 transition-colors"
              >
                ← Anterior
              </button>
              <span className="text-xs text-gray-600">
                {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, data.total)} de {data.total}
              </span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={(page + 1) * PAGE_SIZE >= data.total}
                className="text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30 transition-colors"
              >
                Siguiente →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
