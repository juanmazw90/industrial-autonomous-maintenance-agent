"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { api, type MLModel } from "@/lib/api";
import { FlaskConical, Tag, ChevronRight } from "lucide-react";

const STAGE_CFG: Record<string, string> = {
  Production: "text-green-400 bg-green-950/40 border-green-800/50",
  Staging:    "text-yellow-400 bg-yellow-950/40 border-yellow-800/50",
  Archived:   "text-gray-500 bg-gray-800/40 border-gray-700/50",
  None:       "text-gray-500 bg-gray-800/40 border-gray-700/50",
};

const MODEL_ACCENT: Record<string, string> = {
  "amia-failure-prediction": "border-red-800/40 hover:border-red-700/60",
  "amia-rca-model":          "border-yellow-800/40 hover:border-yellow-700/60",
  "amia-rul-model":          "border-indigo-800/40 hover:border-indigo-700/60",
};

function ModelCard({ m }: { m: MLModel }) {
  const accentBorder = MODEL_ACCENT[m.name] ?? "border-gray-800 hover:border-gray-600";
  const stage = m.latest_stage ?? "None";
  const stageCls = STAGE_CFG[stage] ?? STAGE_CFG.None;

  return (
    <Link href={`/models/${encodeURIComponent(m.name)}`}>
      <div className={`border rounded-xl p-5 bg-gray-900/40 transition-all duration-200 cursor-pointer group ${accentBorder}`}>
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <FlaskConical size={14} className="text-gray-500 shrink-0 mt-0.5" />
            <p className="text-sm font-semibold text-gray-200">{m.name}</p>
          </div>
          <ChevronRight size={14} className="text-gray-600 group-hover:text-indigo-400 transition-colors shrink-0" />
        </div>

        <div className="flex items-center gap-3 mb-3">
          <span className={`text-[11px] font-semibold uppercase px-2 py-0.5 rounded border ${stageCls}`}>
            {stage}
          </span>
          {m.latest_version && (
            <span className="text-[11px] text-gray-500 font-mono">v{m.latest_version}</span>
          )}
        </div>

        {m.description && (
          <p className="text-xs text-gray-600 line-clamp-2 mb-3">{m.description}</p>
        )}

        {Object.keys(m.tags).length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(m.tags).slice(0, 3).map(([k, v]) => (
              <span key={k} className="flex items-center gap-1 text-[11px] bg-gray-800/50 text-gray-500 px-2 py-0.5 rounded">
                <Tag size={9} />
                {k}: {v}
              </span>
            ))}
          </div>
        )}
      </div>
    </Link>
  );
}

export default function ModelsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["models"],
    queryFn: api.models.list,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  return (
    <div className="px-8 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-gray-100">Modelos ML</h1>
        <p className="text-sm text-gray-500 mt-0.5">Modelos registrados en MLflow — versiones, métricas y deriva</p>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 animate-pulse">
          {[...Array(3)].map((_, i) => <div key={i} className="h-32 rounded-xl bg-gray-800/40" />)}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-800/60 bg-red-950/30 px-5 py-4 text-sm text-red-400">
          MLflow no disponible — asegúrese de que MLFLOW_TRACKING_URI está configurado.
        </div>
      )}

      {data && data.length === 0 && (
        <div className="py-16 text-center text-gray-600 text-sm">
          No hay modelos registrados en MLflow.
        </div>
      )}

      {data && data.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {data.map((m) => <ModelCard key={m.name} m={m} />)}
        </div>
      )}
    </div>
  );
}
