"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type AlertItem } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { AlertTriangle, Filter, CheckCircle2 } from "lucide-react";

const SEV_CFG: Record<string, { badge: string; dot: string }> = {
  critical: { badge: "text-red-400 bg-red-950/40 border-red-700/60",     dot: "bg-red-500" },
  high:     { badge: "text-orange-400 bg-orange-950/40 border-orange-700/60", dot: "bg-orange-500" },
  medium:   { badge: "text-yellow-400 bg-yellow-950/40 border-yellow-700/60", dot: "bg-yellow-500" },
  low:      { badge: "text-blue-400 bg-blue-950/40 border-blue-700/60",   dot: "bg-blue-500" },
};

const STATUS_CFG: Record<string, string> = {
  new:          "text-red-400",
  acknowledged: "text-yellow-400",
  assigned:     "text-blue-400",
  resolved:     "text-green-400",
};

const TRANSITIONS: Record<string, string[]> = {
  new:          ["acknowledged", "resolved"],
  acknowledged: ["assigned", "resolved"],
  assigned:     ["resolved"],
  resolved:     [],
};

function SeverityBadge({ severity }: { severity: string }) {
  const cfg = SEV_CFG[severity] ?? SEV_CFG.low;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase px-2.5 py-0.5 rounded-full border ${cfg.badge}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {severity}
    </span>
  );
}

function AlertRow({ alert, onTransition }: { alert: AlertItem; onTransition: (id: string, status: string) => void }) {
  const transitions = TRANSITIONS[alert.status] ?? [];

  return (
    <tr className="hover:bg-gray-900/40 transition-colors border-b border-gray-800/60 last:border-0">
      <td className="px-4 py-3">
        <SeverityBadge severity={alert.severity} />
      </td>
      <td className="px-4 py-3">
        <span className={`text-xs font-medium uppercase ${STATUS_CFG[alert.status] ?? "text-gray-400"}`}>
          {alert.status}
        </span>
      </td>
      <td className="px-4 py-3">
        <p className="text-sm text-gray-200 font-medium leading-tight">{alert.title}</p>
        {alert.description && (
          <p className="text-xs text-gray-600 mt-0.5 line-clamp-1">{alert.description}</p>
        )}
      </td>
      <td className="px-4 py-3">
        <span className="text-xs font-mono text-gray-400">{alert.machine_code}</span>
        <p className="text-[11px] text-gray-600">{alert.machine_name}</p>
      </td>
      <td className="px-4 py-3 text-xs text-gray-500">{alert.source}</td>
      <td className="px-4 py-3 text-xs text-gray-600 whitespace-nowrap">{fmtTs(alert.created_at)}</td>
      <td className="px-4 py-3">
        <div className="flex gap-1.5">
          {transitions.map((t) => (
            <button
              key={t}
              onClick={() => onTransition(alert.id, t)}
              className={`text-[11px] font-medium px-2.5 py-1 rounded-lg border transition-colors ${
                t === "resolved"
                  ? "border-green-800/60 text-green-500 hover:bg-green-950/30"
                  : "border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </td>
    </tr>
  );
}

export default function AlertsPage() {
  const [statusFilter, setStatusFilter] = useState("new");
  const [severityFilter, setSeverityFilter] = useState("");
  const qc = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["alerts", statusFilter, severityFilter],
    queryFn: () =>
      api.alerts.list({
        status: statusFilter || undefined,
        severity: severityFilter || undefined,
        limit: 100,
      }),
    refetchInterval: 15_000,
  });

  const mutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      api.alerts.transition(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });

  const alerts = data?.alerts ?? [];
  const total = data?.total ?? 0;

  const sevCounts = alerts.reduce((acc, a) => {
    acc[a.severity] = (acc[a.severity] ?? 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="px-8 py-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Centro de Alertas</h1>
          <p className="text-sm text-gray-500 mt-0.5">Monitorear y responder alertas del sistema</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          {["critical", "high", "medium", "low"].map((s) => (
            <span key={s} className={`${SEV_CFG[s].badge.split(" ")[0]}`}>
              {s}: <b>{sevCounts[s] ?? 0}</b>
            </span>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-6 flex-wrap">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <Filter size={12} className="text-gray-500" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">Todos los estados</option>
            <option value="new">Nuevo</option>
            <option value="acknowledged">Reconocido</option>
            <option value="assigned">Asignado</option>
            <option value="resolved">Resuelto</option>
          </select>
        </div>
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">Todas las severidades</option>
            <option value="critical">Crítico</option>
            <option value="high">Alto</option>
            <option value="medium">Medio</option>
            <option value="low">Bajo</option>
          </select>
        </div>
        <span className="text-xs text-gray-600 self-center ml-auto">{total} alertas</span>
      </div>

      {error && (
        <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400 mb-4">
          Error al cargar alertas.
        </div>
      )}

      {alerts.length === 0 && !isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <CheckCircle2 size={32} className="text-green-500/40" />
          <p className="text-gray-600 text-sm">Ninguna alerta coincide con el filtro.</p>
        </div>
      ) : (
        <div className="border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900/80 border-b border-gray-800">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Severidad</th>
                <th className="px-4 py-3 text-left">Estado</th>
                <th className="px-4 py-3 text-left">Título</th>
                <th className="px-4 py-3 text-left">Máquina</th>
                <th className="px-4 py-3 text-left">Fuente</th>
                <th className="px-4 py-3 text-left">Creado</th>
                <th className="px-4 py-3 text-left">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-gray-600 text-xs">Cargando…</td>
                </tr>
              ) : (
                alerts.map((a) => (
                  <AlertRow
                    key={a.id}
                    alert={a}
                    onTransition={(id, status) => mutation.mutate({ id, status })}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
