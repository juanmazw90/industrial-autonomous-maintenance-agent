"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type AuditEntry } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { ShieldCheck } from "lucide-react";

const PAGE_SIZE = 50;

const ACTION_LABEL: Record<string, { label: string; color: string }> = {
  post:   { label: "Crear",     color: "text-green-400 bg-green-950/50 border-green-800/60" },
  patch:  { label: "Actualizar", color: "text-blue-400 bg-blue-950/50 border-blue-800/60" },
  put:    { label: "Reemplazar", color: "text-blue-400 bg-blue-950/50 border-blue-800/60" },
  delete: { label: "Eliminar",  color: "text-red-400 bg-red-950/50 border-red-800/60" },
};

function ActionBadge({ action }: { action: string }) {
  const cfg = ACTION_LABEL[action.toLowerCase()] ?? { label: action.toUpperCase(), color: "text-gray-400 bg-gray-800 border-gray-700" };
  return (
    <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border ${cfg.color}`}>
      {cfg.label}
    </span>
  );
}

function DiffPreview({ diff }: { diff: Record<string, unknown> | null }) {
  if (!diff) return <span className="text-gray-700">—</span>;
  const keys = Object.keys(diff).slice(0, 3);
  if (!keys.length) return <span className="text-gray-700">—</span>;
  return (
    <span className="text-gray-500 text-[11px] font-mono truncate max-w-xs block">
      {keys.map((k) => `${k}: ${JSON.stringify(diff[k])}`).join(", ")}
    </span>
  );
}

export default function AuditPage() {
  const [entityType, setEntityType] = useState("");
  const [actorId, setActorId]       = useState("");
  const [offset, setOffset]          = useState(0);

  const { data, isLoading, error } = useQuery({
    queryKey: ["audit", entityType, actorId, offset],
    queryFn: () =>
      api.audit.list({
        entity_type: entityType || undefined,
        actor_id:    actorId    || undefined,
        limit:       PAGE_SIZE,
        offset,
      }),
    staleTime: 10_000,
  });

  function handleFilter(e: React.FormEvent) {
    e.preventDefault();
    setOffset(0);
  }

  return (
    <div className="px-8 py-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100 flex items-center gap-2">
            <ShieldCheck size={18} className="text-indigo-400" />
            Registro de Auditoría
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">Trazabilidad de todas las operaciones de escritura del sistema</p>
        </div>
        {data && (
          <p className="text-xs text-gray-600">{data.total.toLocaleString()} entradas</p>
        )}
      </div>

      {/* Filters */}
      <form onSubmit={handleFilter} className="flex gap-3 mb-6 flex-wrap">
        <input
          type="text"
          placeholder="Tipo de entidad (ej. work-orders)"
          value={entityType}
          onChange={(e) => setEntityType(e.target.value)}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-2 w-52 outline-none focus:border-indigo-600 placeholder-gray-600"
        />
        <input
          type="text"
          placeholder="ID de actor"
          value={actorId}
          onChange={(e) => setActorId(e.target.value)}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-2 w-44 outline-none focus:border-indigo-600 placeholder-gray-600"
        />
        <button
          type="submit"
          className="bg-indigo-700 hover:bg-indigo-600 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors"
        >
          Filtrar
        </button>
        {(entityType || actorId) && (
          <button
            type="button"
            onClick={() => { setEntityType(""); setActorId(""); setOffset(0); }}
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            Limpiar
          </button>
        )}
      </form>

      {/* Table */}
      {isLoading && (
        <div className="space-y-2 animate-pulse">
          {[...Array(8)].map((_, i) => <div key={i} className="h-10 rounded-lg bg-gray-800/50" />)}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400">
          No se puede conectar al backend.
        </div>
      )}

      {data && (
        <>
          <div className="overflow-x-auto rounded-xl border border-gray-800">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-900/80 text-xs text-gray-500 uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3">Fecha / Hora</th>
                  <th className="px-4 py-3">Acción</th>
                  <th className="px-4 py-3">Tipo de Entidad</th>
                  <th className="px-4 py-3">ID Entidad</th>
                  <th className="px-4 py-3">Actor</th>
                  <th className="px-4 py-3">Tipo Actor</th>
                  <th className="px-4 py-3">Resumen</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {data.entries.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-sm">
                      Sin entradas de auditoría con los filtros actuales.
                    </td>
                  </tr>
                )}
                {data.entries.map((entry: AuditEntry) => (
                  <tr key={entry.id} className="hover:bg-gray-900/40 transition-colors">
                    <td className="px-4 py-3 text-gray-500 text-xs font-mono whitespace-nowrap">
                      {fmtTs(entry.created_at, { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </td>
                    <td className="px-4 py-3">
                      <ActionBadge action={entry.action} />
                    </td>
                    <td className="px-4 py-3 text-gray-300 font-mono text-xs">
                      {entry.entity_type}
                    </td>
                    <td className="px-4 py-3 text-gray-500 font-mono text-xs truncate max-w-[120px]">
                      {entry.entity_id ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs font-mono">
                      {entry.actor_id ?? <span className="text-gray-700">—</span>}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                        entry.actor_type === "user"
                          ? "text-indigo-400 bg-indigo-950/50 border-indigo-800/60"
                          : "text-gray-500 bg-gray-800/60 border-gray-700"
                      }`}>
                        {entry.actor_type}
                      </span>
                    </td>
                    <td className="px-4 py-3 max-w-xs">
                      <DiffPreview diff={entry.diff} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <p className="text-xs text-gray-600">
              Mostrando {offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} de {data.total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className="text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-not-allowed border border-gray-700 rounded-lg px-3 py-1.5 transition-colors"
              >
                ← Anterior
              </button>
              <button
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= data.total}
                className="text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-not-allowed border border-gray-700 rounded-lg px-3 py-1.5 transition-colors"
              >
                Siguiente →
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
