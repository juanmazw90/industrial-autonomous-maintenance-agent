"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, type LogEntry, type AuditEntry } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { Search, Filter, FileText } from "lucide-react";

type View = "logs" | "audit";

const LEVEL_CFG: Record<string, string> = {
  critical: "text-red-400 bg-red-950/40 border-red-800/50",
  error:    "text-red-400 bg-red-950/30 border-red-900/50",
  warning:  "text-yellow-400 bg-yellow-950/40 border-yellow-800/50",
  warn:     "text-yellow-400 bg-yellow-950/40 border-yellow-800/50",
  info:     "text-blue-400 bg-blue-950/30 border-blue-900/50",
  debug:    "text-gray-500 bg-gray-800/30 border-gray-700/50",
};

function LevelBadge({ level }: { level?: string }) {
  const l = (level ?? "info").toLowerCase();
  const cls = LEVEL_CFG[l] ?? LEVEL_CFG.info;
  return (
    <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${cls}`}>
      {l}
    </span>
  );
}

function LogsTable() {
  const [levelFilter, setLevelFilter] = useState("");
  const [search, setSearch] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["logs", levelFilter],
    queryFn: () => api.logs.list({ limit: 200, level: levelFilter || undefined }),
    refetchInterval: 15_000,
  });

  const entries = (data?.entries ?? []).filter((e) =>
    !search ||
    JSON.stringify(e).toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-4">
      {data?.source && data.source !== "not_configured" && (
        <p className="text-xs text-gray-600 font-mono">Source: {data.source}</p>
      )}
      {data?.source === "not_configured" && (
        <div className="rounded-xl border border-yellow-800/50 bg-yellow-950/20 px-5 py-4 text-sm text-yellow-400">
          Set <code className="font-mono bg-yellow-950/40 px-1.5 py-0.5 rounded text-xs">LOG_FILE_PATH</code> env variable
          to enable log streaming. Structured JSON logs (structlog) will appear here.
        </div>
      )}

      <div className="flex gap-3 flex-wrap">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm flex-1 min-w-48">
          <Search size={13} className="text-gray-500 shrink-0" />
          <input
            type="text"
            placeholder="Buscar registros…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent outline-none text-gray-200 placeholder-gray-600 w-full text-sm"
          />
        </div>
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <Filter size={12} className="text-gray-500" />
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">Todos los niveles</option>
            <option value="debug">Debug</option>
            <option value="info">Info</option>
            <option value="warning">Advertencia</option>
            <option value="error">Error</option>
            <option value="critical">Crítico</option>
          </select>
        </div>
        <span className="text-xs text-gray-600 self-center">{entries.length} entradas</span>
      </div>

      {entries.length === 0 && !isLoading ? (
        <div className="py-16 text-center text-gray-600 text-sm">
          {data?.source === "not_configured" ? "Configure LOG_FILE_PATH para ver registros aquí." : "Ninguna entrada coincide con el filtro."}
        </div>
      ) : (
        <div className="border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-xs font-mono">
            <thead className="bg-gray-900/80 border-b border-gray-800">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wider font-sans">
                <th className="px-4 py-3 text-left w-36">Marca de Tiempo</th>
                <th className="px-4 py-3 text-left w-20">Nivel</th>
                <th className="px-4 py-3 text-left w-36">Logger</th>
                <th className="px-4 py-3 text-left">Mensaje</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/40">
              {isLoading && (
                <tr><td colSpan={4} className="px-4 py-8 text-center text-gray-600 font-sans">Cargando…</td></tr>
              )}
              {entries.map((e, i) => {
                const ts = e.timestamp as string | undefined;
                const level = e.level as string | undefined;
                const logger = e.logger as string | undefined;
                const msg = (e.event ?? e.message ?? JSON.stringify(e)) as string;
                return (
                  <tr key={`${ts ?? ""}|${logger ?? ""}|${msg.slice(0, 60)}|${i}`} className={`hover:bg-gray-900/30 transition-colors ${
                    level === "error" || level === "critical" ? "bg-red-950/10" :
                    level === "warning" || level === "warn" ? "bg-yellow-950/5" : ""
                  }`}>
                    <td className="px-4 py-2 text-gray-600 whitespace-nowrap text-[11px]">
                      {ts ? fmtTs(ts) : "—"}
                    </td>
                    <td className="px-4 py-2">
                      <LevelBadge level={level} />
                    </td>
                    <td className="px-4 py-2 text-gray-600 text-[11px] truncate max-w-[9rem]">
                      {logger ?? "—"}
                    </td>
                    <td className="px-4 py-2 text-gray-300 text-[11px] break-all">
                      {msg.length > 300 ? msg.slice(0, 300) + "…" : msg}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AuditTable() {
  const [entityFilter, setEntityFilter] = useState("");
  const [search, setSearch] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["audit", entityFilter],
    queryFn: () =>
      api.audit.list({
        entity_type: entityFilter || undefined,
        limit: 100,
      }),
    refetchInterval: 30_000,
  });

  const entries = (data?.entries ?? []).filter((e) =>
    !search || JSON.stringify(e).toLowerCase().includes(search.toLowerCase())
  );

  const entityTypes = [...new Set((data?.entries ?? []).map((e) => e.entity_type))];

  return (
    <div className="space-y-4">
      <div className="flex gap-3 flex-wrap">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm flex-1 min-w-48">
          <Search size={13} className="text-gray-500 shrink-0" />
          <input
            type="text"
            placeholder="Buscar auditoría…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent outline-none text-gray-200 placeholder-gray-600 w-full text-sm"
          />
        </div>
        <select
          value={entityFilter}
          onChange={(e) => setEntityFilter(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 outline-none cursor-pointer"
        >
          <option value="">Todas las entidades</option>
          {entityTypes.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <span className="text-xs text-gray-600 self-center">{data?.total ?? 0} entradas</span>
      </div>

      <div className="border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-900/80 border-b border-gray-800">
            <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-3 text-left">Acción</th>
              <th className="px-4 py-3 text-left">Entidad</th>
              <th className="px-4 py-3 text-left">Actor</th>
              <th className="px-4 py-3 text-left">Diff</th>
              <th className="px-4 py-3 text-left">Hora</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {isLoading && (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-gray-600 text-xs">Cargando…</td></tr>
            )}
            {entries.map((e) => (
              <tr key={e.id} className="hover:bg-gray-900/40 transition-colors">
                <td className="px-4 py-3">
                  <span className={`text-[11px] font-semibold uppercase px-2 py-0.5 rounded border ${
                    e.action === "delete" ? "text-red-400 border-red-900/50" :
                    e.action === "post" || e.action === "create" ? "text-green-400 border-green-900/50" :
                    "text-gray-400 border-gray-700/50"
                  }`}>
                    {e.action}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <p className="text-xs text-gray-400">{e.entity_type}</p>
                  {e.entity_id && <p className="text-[11px] text-gray-600 font-mono">{e.entity_id.slice(0, 12)}</p>}
                </td>
                <td className="px-4 py-3">
                  <p className="text-xs text-gray-400">{e.actor_type}</p>
                  {e.actor_id && <p className="text-[11px] text-gray-600">{e.actor_id.slice(0, 12)}</p>}
                </td>
                <td className="px-4 py-3">
                  {e.diff && (
                    <pre className="text-[11px] text-gray-600 max-w-xs overflow-hidden text-ellipsis">
                      {JSON.stringify(e.diff).slice(0, 80)}
                    </pre>
                  )}
                </td>
                <td className="px-4 py-3 text-xs text-gray-600 whitespace-nowrap">{fmtTs(e.created_at)}</td>
              </tr>
            ))}
            {!isLoading && entries.length === 0 && (
              <tr><td colSpan={5} className="px-4 py-12 text-center text-gray-600 text-sm">Sin entradas de auditoría.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function LogsPage() {
  const [view, setView] = useState<View>("logs");

  return (
    <div className="px-8 py-8 max-w-6xl">
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Registros y Auditoría</h1>
          <p className="text-sm text-gray-500 mt-0.5">Registros estructurados de la aplicación y auditoría de cambios</p>
        </div>
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-lg p-1">
          {(["logs", "audit"] as View[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 text-xs font-medium rounded capitalize transition-colors ${
                view === v ? "bg-gray-800 text-gray-200" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {v === "logs" ? "Registros de App" : "Auditoría"}
            </button>
          ))}
        </div>
      </div>

      {view === "logs"  && <LogsTable />}
      {view === "audit" && <AuditTable />}
    </div>
  );
}
