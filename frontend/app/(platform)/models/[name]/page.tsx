"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { api } from "@/lib/api";
import { ArrowLeft, ExternalLink } from "lucide-react";

type Tab = "metrics" | "drift";

const TAB_LABELS: Record<Tab, string> = {
  metrics: "Métricas",
  drift:   "Deriva",
};

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5 py-4">
      <p className="text-[11px] text-gray-600 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold tabular-nums text-gray-100">{value}</p>
    </div>
  );
}

// Format metric key for display
function fmtKey(k: string) {
  return k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function MetricsTab({ name }: { name: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model-metrics", name],
    queryFn: () => api.models.metrics(name),
    staleTime: 60_000,
  });

  if (isLoading) return <div className="animate-pulse h-64 bg-gray-800/40 rounded-xl" />;
  if (error)
    return <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400">
      No se pudieron cargar las métricas.
    </div>;
  if (!data) return null;

  const metrics = data.metrics;
  const params = data.params;

  // Key metrics tiles (known important ones first)
  const KEY_METRICS = ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall",
                        "rmse", "mae", "r2", "optimal_threshold"];
  const primary = KEY_METRICS.filter((k) => k in metrics);
  const rest = Object.keys(metrics).filter((k) => !KEY_METRICS.includes(k));

  // Feature importance — stored as "feature_importance_{name}" or "fi_{name}"
  const fiEntries = Object.entries(metrics)
    .filter(([k]) => k.startsWith("feature_importance_") || k.startsWith("fi_"))
    .map(([k, v]) => ({
      feature: k.replace(/^feature_importance_|^fi_/, ""),
      value: v,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 15);

  const barColors = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff"];

  return (
    <div className="space-y-6">
      {/* Primary metric tiles */}
      {primary.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {primary.map((k) => (
            <MetricTile
              key={k}
              label={fmtKey(k)}
              value={metrics[k] != null ? metrics[k].toFixed(4) : "—"}
            />
          ))}
        </div>
      )}

      {/* All metrics table */}
      {rest.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900/80 border-b border-gray-800">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Métrica</th>
                <th className="px-4 py-3 text-right">Valor</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60">
              {rest.map((k) => (
                <tr key={k} className="hover:bg-gray-800/20">
                  <td className="px-4 py-2 text-gray-400 font-mono text-xs">{k}</td>
                  <td className="px-4 py-2 text-right text-gray-200 tabular-nums text-xs">
                    {typeof metrics[k] === "number" ? metrics[k].toFixed(6) : String(metrics[k])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Feature importance chart */}
      {fiEntries.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Importancia de Variables (top 15)</h3>
          <ResponsiveContainer width="100%" height={Math.max(200, fiEntries.length * 22)}>
            <BarChart data={fiEntries} layout="vertical" margin={{ left: 16, right: 16 }}>
              <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="feature" tick={{ fill: "#9ca3af", fontSize: 11 }} width={140} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#e5e7eb" }}
                itemStyle={{ color: "#a5b4fc" }}
                formatter={(v) => [(v as number).toFixed(4), "importance"]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {fiEntries.map((_, i) => (
                  <Cell key={i} fill={barColors[i % barColors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Params */}
      {Object.keys(params).length > 0 && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Parámetros de Entrenamiento</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(params).map(([k, v]) => (
              <div key={k} className="text-xs">
                <p className="text-gray-600 font-mono">{k}</p>
                <p className="text-gray-300 font-semibold mt-0.5">{v}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Run link */}
      <p className="text-xs text-gray-600">
        Ejecución MLflow: <span className="font-mono text-gray-500">{data.run_id}</span>
      </p>
    </div>
  );
}

function DriftTab({ name }: { name: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["model-drift", name],
    queryFn: () => api.models.drift(name),
    staleTime: 120_000,
  });

  if (isLoading) return <div className="animate-pulse h-32 bg-gray-800/40 rounded-xl" />;
  if (!data) return null;

  if (data.status === "no_report") {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/40 px-5 py-8 text-center space-y-3">
        <p className="text-sm text-gray-400">Sin informe de deriva disponible aún.</p>
        <p className="text-xs text-gray-600">{data.message}</p>
      </div>
    );
  }

  const drifted = data.dataset_drift ?? false;
  const sharePct = data.share_drifted_features != null
    ? (data.share_drifted_features * 100).toFixed(1)
    : null;

  return (
    <div className="space-y-5">
      {/* Overall drift */}
      <div className={`border rounded-xl px-5 py-4 flex items-center gap-4 ${drifted ? "border-red-700/60 bg-red-950/30" : "border-green-800/50 bg-green-950/20"}`}>
        <div className={`w-3 h-3 rounded-full ${drifted ? "bg-red-500 animate-pulse" : "bg-green-500"}`} />
        <div>
          <p className={`text-sm font-semibold ${drifted ? "text-red-400" : "text-green-400"}`}>
            {drifted ? "Deriva de dataset detectada" : "Sin deriva de dataset"}
          </p>
          {sharePct && (
            <p className="text-xs text-gray-500 mt-0.5">
              {sharePct}% de variables con deriva
              {data.number_of_drifted_columns != null && ` (${data.number_of_drifted_columns} / ${data.number_of_columns})`}
            </p>
          )}
        </div>
        <a
          href="/api/drift-report"
          target="_blank"
          rel="noopener"
          className="ml-auto flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 border border-indigo-900/60 px-3 py-1.5 rounded-lg"
        >
          Informe completo <ExternalLink size={11} />
        </a>
      </div>

      {/* Per-feature drift table */}
      {data.feature_drift && Object.keys(data.feature_drift).length > 0 && (
        <div className="border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900/80 border-b border-gray-800">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Variable</th>
                <th className="px-4 py-3 text-left">Prueba</th>
                <th className="px-4 py-3 text-right">p-value</th>
                <th className="px-4 py-3 text-center">Deriva</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60">
              {Object.entries(data.feature_drift)
                .sort(([, a], [, b]) => (a.p_value ?? 1) - (b.p_value ?? 1))
                .map(([feat, info]) => (
                <tr key={feat} className={`hover:bg-gray-800/20 ${info.drift_detected ? "bg-red-950/10" : ""}`}>
                  <td className="px-4 py-2 text-gray-300 text-xs font-mono">{feat}</td>
                  <td className="px-4 py-2 text-gray-500 text-xs">{info.stattest}</td>
                  <td className="px-4 py-2 text-right tabular-nums text-xs text-gray-400">
                    {info.p_value?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-4 py-2 text-center">
                    <span className={`text-[10px] font-semibold uppercase ${info.drift_detected ? "text-red-400" : "text-green-500"}`}>
                      {info.drift_detected ? "YES" : "no"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data.generated_at && (
        <p className="text-xs text-gray-600">Informe generado: {data.generated_at}</p>
      )}
    </div>
  );
}

export default function ModelDetailPage({ params }: { params: { name: string } }) {
  const name = decodeURIComponent(params.name);
  const [tab, setTab] = useState<Tab>("metrics");

  return (
    <div className="px-8 py-8 max-w-4xl">
      <div className="flex items-center gap-3 mb-6">
        <Link href="/models" className="text-gray-600 hover:text-gray-300 transition-colors">
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-xl font-semibold text-gray-100 font-mono">{name}</h1>
          <p className="text-sm text-gray-500">Detalles del modelo MLflow</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-800 mb-6">
        {(["metrics", "drift"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition-colors ${
              tab === t
                ? "border-indigo-500 text-indigo-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      {tab === "metrics" && <MetricsTab name={name} />}
      {tab === "drift"   && <DriftTab name={name} />}
    </div>
  );
}
