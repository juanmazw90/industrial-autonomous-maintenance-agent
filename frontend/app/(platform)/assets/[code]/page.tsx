"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api, type TimelineEvent } from "@/lib/api";
import { fmt, fmtTs, fmtDuration } from "@/lib/utils";
import { ArrowLeft, AlertTriangle, Wrench, Clock, BarChart3 } from "lucide-react";

const STATUS_CFG = {
  healthy:  { color: "text-green-400",  bg: "bg-green-950/40 border-green-800/60" },
  warning:  { color: "text-yellow-400", bg: "bg-yellow-950/40 border-yellow-700/60" },
  critical: { color: "text-red-400",    bg: "bg-red-950/40 border-red-700/60" },
};

const KIND_CFG: Record<string, { icon: string; color: string }> = {
  sensor_anomaly:  { icon: "⚠", color: "text-orange-400" },
  prediction:      { icon: "◆", color: "text-indigo-400" },
  agent_decision:  { icon: "▣", color: "text-purple-400" },
  rca:             { icon: "◉", color: "text-yellow-400" },
  economic:        { icon: "€", color: "text-teal-400" },
  wo_created:      { icon: "✔", color: "text-green-400" },
  alert_created:   { icon: "!", color: "text-red-400" },
  alert_resolved:  { icon: "✓", color: "text-green-500" },
};

type Tab = "overview" | "timeline" | "work_orders";

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "overview",    label: "Overview" },
    { id: "timeline",    label: "Timeline" },
    { id: "work_orders", label: "Work Orders" },
  ];
  return (
    <div className="flex gap-1 border-b border-gray-800 mb-6">
      {tabs.map(({ id, label }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition-colors ${
            active === id
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-gray-500 hover:text-gray-300"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function OverviewTab({ code }: { code: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["asset", code],
    queryFn: () => api.assets.get(code),
    refetchInterval: 20_000,
  });

  if (isLoading) return <div className="animate-pulse h-64 bg-gray-800/40 rounded-xl" />;
  if (!data) return null;

  const cfg = STATUS_CFG[data.status as keyof typeof STATUS_CFG] ?? STATUS_CFG.healthy;
  const pred = data.latest_prediction;
  const probPct = pred?.probability != null ? Math.round(pred.probability * 100) : null;

  return (
    <div className="space-y-6">
      {/* Machine info */}
      <div className={`border rounded-xl p-6 flex flex-col sm:flex-row gap-6 ${cfg.bg}`}>
        <div className="flex-1">
          <h2 className="text-lg font-bold text-gray-100">{data.name}</h2>
          <p className="text-sm text-gray-500 mt-0.5 capitalize">{data.type.replace(/_/g, " ")}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-sm font-semibold uppercase ${cfg.color}`}>{data.status}</span>
          {probPct != null && (
            <div className="text-right">
              <p className={`text-3xl font-bold tabular-nums ${probPct >= 70 ? "text-red-400" : probPct >= 35 ? "text-yellow-400" : "text-green-400"}`}>
                {probPct}%
              </p>
              <p className="text-xs text-gray-600">failure risk</p>
            </div>
          )}
        </div>
      </div>

      {/* Latest prediction */}
      {pred && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 space-y-4">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-2">
            <BarChart3 size={11} /> Latest Prediction — {pred.model_name}
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-xs text-gray-600">Failure probability</p>
              <p className={`text-xl font-bold tabular-nums mt-0.5 ${probPct != null && probPct >= 70 ? "text-red-400" : probPct != null && probPct >= 35 ? "text-yellow-400" : "text-green-400"}`}>
                {probPct != null ? `${probPct}%` : "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Predicted at</p>
              <p className="text-gray-300 mt-0.5 text-sm">{fmtTs(pred.created_at)}</p>
            </div>
          </div>
          {/* SHAP top features */}
          {Array.isArray(pred.shap_top_features) && pred.shap_top_features.length > 0 && (
            <div>
              <p className="text-xs text-gray-600 mb-2">Top contributing features (SHAP)</p>
              <div className="space-y-1.5">
                {(pred.shap_top_features as Array<{ feature: string; value: number }>).slice(0, 5).map((f) => {
                  const abs = Math.abs(f.value);
                  const max = 0.3;
                  const pct = Math.min((abs / max) * 100, 100);
                  return (
                    <div key={f.feature} className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-36 truncate shrink-0">{f.feature}</span>
                      <div className="flex-1 h-1.5 bg-gray-800 rounded-full">
                        <div
                          className={`h-1.5 rounded-full ${f.value > 0 ? "bg-red-500" : "bg-green-500"}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs tabular-nums text-gray-400 w-12 text-right">
                        {f.value > 0 ? "+" : ""}{f.value.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Open alerts */}
      {data.open_alerts.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <AlertTriangle size={11} /> Open Alerts ({data.open_alerts.length})
          </h3>
          <div className="space-y-2">
            {data.open_alerts.map((a) => (
              <div key={a.id} className="flex items-center justify-between text-sm py-1.5 border-b border-gray-800/50 last:border-0">
                <span className="text-gray-300">{a.title}</span>
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-medium uppercase ${a.severity === "critical" ? "text-red-400" : a.severity === "high" ? "text-orange-400" : "text-yellow-400"}`}>
                    {a.severity}
                  </span>
                  <span className="text-xs text-gray-600">{fmtTs(a.created_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TimelineTab({ code }: { code: string }) {
  const [kindFilter, setKindFilter] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["asset-timeline", code, kindFilter],
    queryFn: () => api.assets.timeline(code, kindFilter || undefined, 100),
    refetchInterval: 30_000,
  });

  if (isLoading) return <div className="animate-pulse h-64 bg-gray-800/40 rounded-xl" />;

  const events = data?.events ?? [];

  const kinds = [...new Set(events.map((e) => e.kind))];

  return (
    <div>
      {/* Filter bar */}
      <div className="flex items-center gap-3 mb-5">
        <select
          value={kindFilter}
          onChange={(e) => setKindFilter(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-sm text-gray-300 outline-none cursor-pointer"
        >
          <option value="">All events ({data?.total ?? 0})</option>
          {kinds.map((k) => (
            <option key={k} value={k}>{k.replace(/_/g, " ")}</option>
          ))}
        </select>
      </div>

      {events.length === 0 ? (
        <div className="py-16 text-center text-gray-600 text-sm">No timeline events.</div>
      ) : (
        <div className="relative pl-6">
          {/* Vertical line */}
          <div className="absolute left-2.5 top-0 bottom-0 w-px bg-gray-800" />

          <div className="space-y-4">
            {events.map((e) => {
              const cfg = KIND_CFG[e.kind] ?? { icon: "·", color: "text-gray-400" };
              return (
                <div key={e.id} className="relative flex gap-4 group">
                  {/* Dot */}
                  <div className={`absolute -left-6 mt-0.5 w-5 h-5 rounded-full bg-gray-900 border border-gray-700 flex items-center justify-center text-[10px] ${cfg.color}`}>
                    {cfg.icon}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0 pb-4 border-b border-gray-800/40 group-last:border-0">
                    <div className="flex items-baseline gap-3 mb-0.5">
                      <span className="text-[11px] text-gray-600 shrink-0">{fmtTs(e.ts)}</span>
                      <span className={`text-[11px] font-medium uppercase tracking-wider ${cfg.color}`}>
                        {e.kind.replace(/_/g, " ")}
                      </span>
                    </div>
                    <p className="text-sm text-gray-300">{e.title}</p>
                    {e.payload && Object.keys(e.payload).length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-2">
                        {Object.entries(e.payload).slice(0, 4).map(([k, v]) => (
                          <span key={k} className="text-[11px] bg-gray-800/60 text-gray-500 px-2 py-0.5 rounded">
                            {k}: <span className="text-gray-400">{typeof v === "number" ? v.toFixed(2) : String(v)}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function WorkOrdersTab({ code }: { code: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["asset", code],
    queryFn: () => api.assets.get(code),
    refetchInterval: 20_000,
  });

  const orders = data?.open_work_orders ?? [];
  if (isLoading) return <div className="animate-pulse h-32 bg-gray-800/40 rounded-xl" />;

  return orders.length === 0 ? (
    <div className="py-16 text-center text-gray-600 text-sm">No open work orders.</div>
  ) : (
    <div className="space-y-3">
      {orders.map((wo) => (
        <div key={wo.id} className="bg-gray-900/60 border border-gray-800 rounded-xl px-5 py-4 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-200 truncate">{wo.title}</p>
            <p className="text-xs text-gray-600 mt-0.5 font-mono">{wo.id.slice(0, 8)}</p>
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <span className={`text-xs font-semibold uppercase ${wo.priority === "critical" ? "text-red-400" : wo.priority === "high" ? "text-orange-400" : "text-yellow-400"}`}>
              {wo.priority}
            </span>
            {wo.estimated_cost != null && (
              <span className="text-xs text-gray-400 tabular-nums">{fmt(wo.estimated_cost, "currency")}</span>
            )}
            <span className={`text-xs px-2 py-0.5 rounded border ${wo.status === "completed" ? "border-green-800 text-green-400" : "border-gray-700 text-gray-400"}`}>
              {wo.status}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function AssetDetailPage({ params }: { params: { code: string } }) {
  const code = params.code.toUpperCase();
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div className="px-8 py-8 max-w-4xl">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/assets" className="text-gray-600 hover:text-gray-300 transition-colors">
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-xl font-semibold text-gray-100">{code}</h1>
          <p className="text-sm text-gray-500">Asset Detail</p>
        </div>
      </div>

      <TabBar active={tab} onChange={setTab} />

      {tab === "overview"    && <OverviewTab code={code} />}
      {tab === "timeline"    && <TimelineTab code={code} />}
      {tab === "work_orders" && <WorkOrdersTab code={code} />}
    </div>
  );
}
