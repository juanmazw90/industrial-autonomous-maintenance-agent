"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api, type TraceNode, type ToolCall } from "@/lib/api";
import { fmtTs, fmtDuration } from "@/lib/utils";
import { ArrowLeft, ChevronDown, ChevronRight, Wrench, Bot } from "lucide-react";

// ── Trace node (recursive) ────────────────────────────────────────────────────

function ToolCallRow({ tc }: { tc: { id: string; tool_name: string; status: string; latency_ms: number | null; started_at: string } }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-l border-gray-800 pl-4 ml-4">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 py-1.5 text-sm w-full text-left hover:text-gray-200 transition-colors group"
      >
        <Wrench size={11} className="text-gray-600 shrink-0" />
        <span className="text-gray-400 font-mono text-xs">{tc.tool_name}</span>
        <span className={`ml-1 text-[10px] font-semibold uppercase ${tc.status === "success" ? "text-green-500" : "text-red-400"}`}>
          {tc.status}
        </span>
        <span className="text-[11px] text-gray-600 ml-auto">{fmtDuration(tc.latency_ms)}</span>
      </button>
    </div>
  );
}

function TraceTreeNode({ node, depth = 0 }: { node: TraceNode; depth?: number }) {
  const [collapsed, setCollapsed] = useState(depth > 1);
  const hasChildren = node.children.length > 0 || node.tool_calls.length > 0;
  const statusColor = node.status === "success" ? "text-green-400" : node.status === "error" ? "text-red-400" : "text-yellow-400";

  return (
    <div className={depth > 0 ? "border-l border-gray-800 pl-4 ml-4" : ""}>
      <div className="flex items-center gap-2 py-2 group">
        {hasChildren ? (
          <button
            onClick={() => setCollapsed((v) => !v)}
            className="text-gray-600 hover:text-gray-300 transition-colors shrink-0"
          >
            {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
          </button>
        ) : (
          <span className="w-[13px] shrink-0" />
        )}
        <Bot size={13} className="text-gray-500 shrink-0" />
        <span className="text-sm font-medium text-gray-200 capitalize">{node.agent_name.replace(/_/g, " ")}</span>
        <span className={`text-[10px] font-semibold uppercase ${statusColor}`}>{node.status}</span>
        <span className="text-xs text-gray-600 ml-auto">{fmtDuration(node.latency_ms)}</span>
        {node.cost_usd != null && (
          <span className="text-[11px] text-gray-600 ml-2">${node.cost_usd.toFixed(5)}</span>
        )}
      </div>

      {!collapsed && (
        <div>
          {node.tool_calls.map((tc) => (
            <ToolCallRow key={tc.id} tc={tc} />
          ))}
          {node.children.map((child) => (
            <TraceTreeNode key={child.id} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Detail panel (run metadata) ───────────────────────────────────────────────

function RunMetadata({ id }: { id: string }) {
  const { data } = useQuery({
    queryKey: ["agent-run", id],
    queryFn: () => api.agents.run(id),
  });

  if (!data) return null;

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 space-y-4 mb-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-xs text-gray-600">Sesión</p>
          <p className="font-mono text-gray-400 text-xs mt-0.5 truncate">{data.session_id}</p>
        </div>
        <div>
          <p className="text-xs text-gray-600">Modelo</p>
          <p className="text-gray-300 text-xs mt-0.5">{data.model ?? "—"}</p>
        </div>
        <div>
          <p className="text-xs text-gray-600">Tokens (entrada / salida)</p>
          <p className="tabular-nums text-gray-300 text-xs mt-0.5">
            {data.input_tokens ?? 0} in / {data.output_tokens ?? 0} out
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-600">Costo</p>
          <p className="tabular-nums text-gray-300 text-xs mt-0.5">
            {data.cost_usd != null ? `$${data.cost_usd.toFixed(5)}` : "—"}
          </p>
        </div>
      </div>
      {data.input_summary && (
        <div>
          <p className="text-xs text-gray-600 mb-1">Entrada</p>
          <p className="text-sm text-gray-400 bg-gray-800/40 rounded-lg px-3 py-2 leading-relaxed">{data.input_summary}</p>
        </div>
      )}
      {data.output_summary && (
        <div>
          <p className="text-xs text-gray-600 mb-1">Salida</p>
          <p className="text-sm text-gray-300 bg-gray-800/40 rounded-lg px-3 py-2 leading-relaxed">{data.output_summary}</p>
        </div>
      )}
      {data.tool_calls.length > 0 && (
        <div>
          <p className="text-xs text-gray-600 mb-2">Llamadas a Herramientas ({data.tool_calls.length})</p>
          <div className="space-y-2">
            {data.tool_calls.map((tc) => (
              <div key={tc.id} className="bg-gray-800/40 rounded-lg px-3 py-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-indigo-400">{tc.tool_name}</span>
                  <div className="flex items-center gap-3">
                    <span className={`text-[10px] font-semibold uppercase ${tc.status === "success" ? "text-green-500" : "text-red-400"}`}>{tc.status}</span>
                    <span className="text-[11px] text-gray-600">{fmtDuration(tc.latency_ms)}</span>
                  </div>
                </div>
                {tc.output && (
                  <pre className="text-[11px] text-gray-500 overflow-x-auto mt-1">
                    {JSON.stringify(tc.output, null, 2).slice(0, 300)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function RunTracePage({ params }: { params: { id: string } }) {
  const { id } = params;

  const { data: trace, isLoading } = useQuery({
    queryKey: ["agent-trace", id],
    queryFn: () => api.agents.trace(id),
  });

  return (
    <div className="px-8 py-8 max-w-4xl">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/agents" className="text-gray-600 hover:text-gray-300 transition-colors">
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Explorador de Trazas</h1>
          <p className="text-xs text-gray-600 font-mono mt-0.5">{id}</p>
        </div>
      </div>

      <RunMetadata id={id} />

      {/* Trace tree */}
      <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5">
        <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Árbol de Ejecución</h2>
        {isLoading && <div className="animate-pulse h-32 bg-gray-800/40 rounded-lg" />}
        {trace && <TraceTreeNode node={trace.trace} depth={0} />}
        {!isLoading && !trace && (
          <p className="text-sm text-gray-600 text-center py-8">Traza no disponible.</p>
        )}
      </div>
    </div>
  );
}
