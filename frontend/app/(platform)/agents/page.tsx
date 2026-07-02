"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api, type AgentRun } from "@/lib/api";
import { fmtTs, fmtDuration } from "@/lib/utils";
import { Bot, ChevronRight, Filter } from "lucide-react";

const AGENT_COLORS: Record<string, string> = {
  supervisor:       "text-purple-400 bg-purple-950/40 border-purple-800/50",
  sensor_analyst:   "text-blue-400 bg-blue-950/40 border-blue-800/50",
  doc_expert:       "text-teal-400 bg-teal-950/40 border-teal-800/50",
  synthesizer:      "text-green-400 bg-green-950/40 border-green-800/50",
  economic_analyst: "text-orange-400 bg-orange-950/40 border-orange-800/50",
  work_order_creator: "text-yellow-400 bg-yellow-950/40 border-yellow-800/50",
  rul_analyst:      "text-indigo-400 bg-indigo-950/40 border-indigo-800/50",
};

function agentColor(name: string) {
  return AGENT_COLORS[name] ?? "text-gray-400 bg-gray-900/40 border-gray-700/50";
}

function AgentCard({
  agent_name, total_calls, success_rate, latency_p50_ms, latency_p95_ms, cost_usd_total,
}: {
  agent_name: string; total_calls: number; success_rate: number;
  latency_p50_ms: number | null; latency_p95_ms: number | null; cost_usd_total: number;
}) {
  const colorClass = agentColor(agent_name);
  return (
    <div className={`border rounded-xl p-5 flex flex-col gap-3 ${colorClass}`}>
      <div className="flex items-center gap-2">
        <Bot size={13} />
        <p className="text-sm font-semibold capitalize">{agent_name.replace(/_/g, " ")}</p>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs mt-1">
        <div>
          <p className="text-gray-600">Calls</p>
          <p className="font-bold tabular-nums text-gray-100 text-base">{total_calls}</p>
        </div>
        <div>
          <p className="text-gray-600">Success</p>
          <p className="font-bold tabular-nums text-gray-100 text-base">{(success_rate * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p className="text-gray-600">p50 latency</p>
          <p className="font-medium text-gray-300">{fmtDuration(latency_p50_ms)}</p>
        </div>
        <div>
          <p className="text-gray-600">p95 latency</p>
          <p className="font-medium text-gray-300">{fmtDuration(latency_p95_ms)}</p>
        </div>
      </div>
      <p className="text-[11px] text-gray-500 border-t border-current/10 pt-2 mt-auto">
        Total cost: <span className="font-mono">${cost_usd_total.toFixed(4)}</span>
      </p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cfgMap: Record<string, string> = {
    success: "text-green-400 bg-green-950/40 border-green-800/50",
    error:   "text-red-400 bg-red-950/40 border-red-800/50",
    running: "text-yellow-400 bg-yellow-950/40 border-yellow-800/50",
  };
  const cls = cfgMap[status] ?? "text-gray-400 bg-gray-800/40 border-gray-700/50";
  return (
    <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${cls}`}>
      {status}
    </span>
  );
}

function RunsTable() {
  const [agentFilter, setAgentFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(0);
  const limit = 20;

  const { data, isLoading } = useQuery({
    queryKey: ["agent-runs", agentFilter, statusFilter, page],
    queryFn: () =>
      api.agents.runs({
        agent: agentFilter || undefined,
        status: statusFilter || undefined,
        limit,
        offset: page * limit,
      }),
    refetchInterval: 20_000,
  });

  const runs = data?.runs ?? [];
  const total = data?.total ?? 0;
  const pages = Math.ceil(total / limit);

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-sm">
          <Filter size={12} className="text-gray-500" />
          <select
            value={agentFilter}
            onChange={(e) => { setAgentFilter(e.target.value); setPage(0); }}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">All agents</option>
            {Object.keys(AGENT_COLORS).map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-sm">
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
            className="bg-transparent text-gray-300 outline-none cursor-pointer"
          >
            <option value="">All statuses</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
            <option value="running">Running</option>
          </select>
        </div>
        <span className="text-xs text-gray-600 ml-auto">{total} total runs</span>
      </div>

      {/* Table */}
      <div className="border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-900/80 border-b border-gray-800">
            <tr className="text-xs text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-3 text-left">Agent</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-right">Latency</th>
              <th className="px-4 py-3 text-right">Tokens</th>
              <th className="px-4 py-3 text-right">Cost</th>
              <th className="px-4 py-3 text-left">Started</th>
              <th className="px-4 py-3 text-center">Trace</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {isLoading && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-xs">Loading…</td>
              </tr>
            )}
            {runs.map((run) => (
              <tr key={run.id} className="hover:bg-gray-900/40 transition-colors group">
                <td className="px-4 py-3">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded border ${agentColor(run.agent_name)}`}>
                    {run.agent_name}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={run.status} />
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-gray-400 text-xs">
                  {fmtDuration(run.latency_ms)}
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-gray-500 text-xs">
                  {(run.input_tokens ?? 0) + (run.output_tokens ?? 0) || "—"}
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-gray-500 text-xs">
                  {run.cost_usd != null ? `$${run.cost_usd.toFixed(5)}` : "—"}
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs">{fmtTs(run.started_at)}</td>
                <td className="px-4 py-3 text-center">
                  <Link
                    href={`/agents/runs/${run.id}`}
                    className="text-indigo-500 hover:text-indigo-300 transition-colors"
                  >
                    <ChevronRight size={14} className="inline" />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1.5 text-xs rounded-lg border border-gray-800 text-gray-500 hover:text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-xs text-gray-600">{page + 1} / {pages}</span>
          <button
            onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
            disabled={page >= pages - 1}
            className="px-3 py-1.5 text-xs rounded-lg border border-gray-800 text-gray-500 hover:text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

export default function AgentsPage() {
  const { data: summary } = useQuery({
    queryKey: ["agents-summary"],
    queryFn: api.agents.summary,
    refetchInterval: 20_000,
  });

  return (
    <div className="px-8 py-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-gray-100">AI Control Center</h1>
        <p className="text-sm text-gray-500 mt-0.5">Agent performance, latency, token usage &amp; cost</p>
      </div>

      {/* Totals bar */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
          {[
            { label: "Total Runs",     value: String(summary.totals.total_calls) },
            { label: "Success Rate",   value: `${(summary.totals.success_rate * 100).toFixed(1)}%` },
            { label: "p50 Latency",   value: fmtDuration(summary.totals.latency_p50_ms) },
            { label: "p95 Latency",   value: fmtDuration(summary.totals.latency_p95_ms) },
            { label: "Total Cost",     value: `$${summary.totals.cost_usd_total.toFixed(4)}` },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3">
              <p className="text-[11px] text-gray-600 uppercase tracking-wider">{label}</p>
              <p className="text-lg font-bold text-gray-100 tabular-nums mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Agent cards */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
          {summary.by_agent.map((a) => (
            <AgentCard key={a.agent_name} {...a} />
          ))}
        </div>
      )}

      {/* Runs table */}
      <div>
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Recent Runs</h2>
        <RunsTable />
      </div>
    </div>
  );
}
