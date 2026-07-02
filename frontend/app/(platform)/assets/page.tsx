"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api, type FleetMachine } from "@/lib/api";
import { fmt, fmtTs } from "@/lib/utils";
import { Search, AlertTriangle, ChevronRight, Filter } from "lucide-react";

const STATUS_CFG = {
  healthy:  { dot: "bg-green-400",  badge: "text-green-400 bg-green-950/50 border-green-800/60", label: "Healthy" },
  warning:  { dot: "bg-yellow-400", badge: "text-yellow-400 bg-yellow-950/50 border-yellow-700/60", label: "Warning" },
  critical: { dot: "bg-red-500 animate-pulse", badge: "text-red-400 bg-red-950/50 border-red-700/60", label: "Critical" },
};

function MachineCard({ m }: { m: FleetMachine }) {
  const cfg = STATUS_CFG[m.status] ?? STATUS_CFG.healthy;
  const prob = m.failure_probability;
  const probPct = prob != null ? Math.round(prob * 100) : null;
  const barColor = probPct == null ? "bg-gray-700" : probPct >= 70 ? "bg-red-500" : probPct >= 35 ? "bg-yellow-500" : "bg-green-500";

  return (
    <Link href={`/assets/${m.code}`} className="block">
      <div className="group border border-gray-800 rounded-xl p-5 bg-gray-900/40 hover:border-gray-600 hover:bg-gray-900/70 transition-all duration-200 flex flex-col gap-3 h-full">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-bold text-gray-100 leading-none">{m.code}</p>
            <p className="text-xs text-gray-500 mt-0.5">{m.name}</p>
          </div>
          <span className={`flex items-center gap-1.5 text-[11px] font-semibold px-2 py-1 rounded-full border ${cfg.badge}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
            {cfg.label}
          </span>
        </div>

        {/* Failure probability bar */}
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <span className="text-[11px] text-gray-600">Failure risk</span>
            <span className={`text-lg font-bold tabular-nums ${probPct != null && probPct >= 70 ? "text-red-400" : probPct != null && probPct >= 35 ? "text-yellow-400" : "text-green-400"}`}>
              {probPct != null ? `${probPct}%` : "—"}
            </span>
          </div>
          <div className="h-1 w-full bg-gray-800 rounded-full">
            <div
              className={`h-1 rounded-full transition-all duration-700 ${barColor}`}
              style={{ width: probPct != null ? `${Math.min(probPct, 100)}%` : "0%" }}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between text-[11px] text-gray-600 mt-auto">
          <span>{m.line.name}</span>
          <div className="flex items-center gap-3">
            {m.open_alerts > 0 && (
              <span className="flex items-center gap-1 text-red-400">
                <AlertTriangle size={10} />
                {m.open_alerts}
              </span>
            )}
            <span className="group-hover:text-indigo-400 transition-colors flex items-center gap-0.5">
              Detail <ChevronRight size={11} />
            </span>
          </div>
        </div>
      </div>
    </Link>
  );
}

function Skeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 animate-pulse">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="h-36 rounded-xl bg-gray-800/40" />
      ))}
    </div>
  );
}

export default function FleetPage() {
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [search, setSearch] = useState("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["fleet", statusFilter],
    queryFn: () => api.fleet.list(statusFilter || undefined),
    refetchInterval: 15_000,
  });

  const filtered = (data ?? []).filter((m) =>
    !search || m.code.toLowerCase().includes(search.toLowerCase()) || m.name.toLowerCase().includes(search.toLowerCase())
  );

  const counts = (data ?? []).reduce(
    (acc, m) => { acc[m.status] = (acc[m.status] ?? 0) + 1; return acc; },
    {} as Record<string, number>
  );

  return (
    <div className="px-8 py-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Fleet</h1>
          <p className="text-sm text-gray-500 mt-0.5">All monitored machines with live predictions</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="text-green-400">● {counts.healthy ?? 0} healthy</span>
          <span className="text-yellow-400">● {counts.warning ?? 0} warning</span>
          <span className="text-red-400">● {counts.critical ?? 0} critical</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-6 flex-wrap">
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm flex-1 min-w-48">
          <Search size={14} className="text-gray-500 shrink-0" />
          <input
            type="text"
            placeholder="Search machines…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent outline-none text-gray-200 placeholder-gray-600 w-full"
          />
        </div>
        <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm">
          <Filter size={14} className="text-gray-500" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-transparent outline-none text-gray-300 cursor-pointer"
          >
            <option value="">All status</option>
            <option value="healthy">Healthy</option>
            <option value="warning">Warning</option>
            <option value="critical">Critical</option>
          </select>
        </div>
      </div>

      {isLoading && <Skeleton />}
      {error && (
        <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400">
          Failed to load fleet data.
        </div>
      )}
      {filtered.length === 0 && !isLoading && !error && (
        <div className="flex flex-col items-center justify-center py-20 text-gray-600 text-sm gap-2">
          <span>No machines match your filter.</span>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((m) => (
          <MachineCard key={m.id} m={m} />
        ))}
      </div>
    </div>
  );
}
