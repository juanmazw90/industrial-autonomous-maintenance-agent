"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { CheckCircle2, XCircle, Clock, Info } from "lucide-react";

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-4">
      <h2 className="text-sm font-semibold text-gray-300">{title}</h2>
      {subtitle && <p className="text-xs text-gray-600 mt-0.5">{subtitle}</p>}
    </div>
  );
}

function ConfigRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-800/60 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className={`text-sm ${mono ? "font-mono text-gray-400" : "text-gray-300 font-medium"}`}>{value}</span>
    </div>
  );
}

function StatusRow({ label, ok, detail }: { label: string; ok: boolean | null; detail?: string }) {
  const Icon = ok === null ? Clock : ok ? CheckCircle2 : XCircle;
  const color = ok === null ? "text-gray-500" : ok ? "text-green-400" : "text-red-400";
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-800/60 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <div className="flex items-center gap-2">
        {detail && <span className="text-xs text-gray-600">{detail}</span>}
        <Icon size={14} className={color} />
      </div>
    </div>
  );
}

function AdrNote({ code, title, decision }: { code: string; title: string; decision: string }) {
  return (
    <div className="flex gap-3 py-3 border-b border-gray-800/60 last:border-0">
      <span className="text-[11px] font-mono text-gray-600 bg-gray-800/60 px-2 py-0.5 rounded shrink-0 self-start mt-0.5">
        {code}
      </span>
      <div>
        <p className="text-sm text-gray-400 font-medium">{title}</p>
        <p className="text-xs text-gray-600 mt-0.5">{decision}</p>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const { data: services } = useQuery({
    queryKey: ["monitoring-services"],
    queryFn: api.monitoring.services,
    refetchInterval: 60_000,
  });

  const { data: ops } = useQuery({
    queryKey: ["operations-summary"],
    queryFn: api.operations.summary,
    staleTime: 30_000,
  });

  const svcMap = Object.fromEntries(
    (services?.services ?? []).map((s) => [s.name, s])
  );

  return (
    <div className="px-8 py-8 max-w-3xl space-y-10">
      <div>
        <h1 className="text-xl font-semibold text-gray-100">Settings</h1>
        <p className="text-sm text-gray-500 mt-0.5">Platform configuration &amp; architecture decisions</p>
      </div>

      {/* ── Platform info ─────────────────────────────────────────────────── */}
      <section>
        <SectionHeader title="Platform" subtitle="Runtime information" />
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5">
          <ConfigRow label="Backend API"      value="v2 (0.7.0)" />
          <ConfigRow label="API Base URL"     value="/api/v2" mono />
          <ConfigRow label="Frontend"         value="Next.js 14 App Router" />
          <ConfigRow label="Data Refresh"     value="15 s polling (TanStack Query)" />
          <ConfigRow label="Fleet Size"       value={ops ? `${ops.machines.total} machines` : "—"} />
          <ConfigRow label="Risk Exposure"    value={ops ? `$${ops.risk_exposure_usd.toLocaleString()}` : "—"} />
        </div>
      </section>

      {/* ── Services ─────────────────────────────────────────────────────── */}
      <section>
        <SectionHeader title="Services" subtitle="Infrastructure health — auto-refreshes every 60 s" />
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5">
          <StatusRow
            label="PostgreSQL"
            ok={svcMap["postgres"]?.status === "ok" ? true : svcMap["postgres"] ? false : null}
            detail={svcMap["postgres"]?.detail}
          />
          <StatusRow
            label="Redis"
            ok={svcMap["redis"]?.status === "ok" ? true : svcMap["redis"] ? false : null}
            detail={svcMap["redis"]?.detail}
          />
          <StatusRow
            label="Qdrant"
            ok={svcMap["qdrant"]?.status === "ok" ? true : svcMap["qdrant"] ? false : null}
            detail={svcMap["qdrant"]?.detail}
          />
        </div>
      </section>

      {/* ── AI / ML ──────────────────────────────────────────────────────── */}
      <section>
        <SectionHeader title="AI / ML Configuration" subtitle="Model serving and observability" />
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5">
          <ConfigRow label="ML Tracking"     value="MLflow (SQLite backend)" />
          <ConfigRow label="Models"          value="amia-failure-prediction · amia-rca-model · amia-rul-model" />
          <ConfigRow label="Embeddings"      value="Qdrant vector store" />
          <ConfigRow label="LLM Agent"       value="LangGraph multi-agent supervisor" />
          <ConfigRow label="Observability"   value="Langfuse (optional — LANGFUSE_PUBLIC_KEY)" />
          <ConfigRow label="Rate Limiting"   value="10 req / min per IP (slowapi)" />
        </div>
      </section>

      {/* ── API versioning ───────────────────────────────────────────────── */}
      <section>
        <SectionHeader title="API Versioning" />
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5">
          <div className="py-3 border-b border-gray-800/60 flex items-center justify-between">
            <span className="text-sm text-gray-500">v2 endpoints</span>
            <span className="text-xs font-semibold text-green-400 bg-green-950/40 border border-green-800/50 px-2 py-0.5 rounded uppercase">Active</span>
          </div>
          <div className="py-3 border-b border-gray-800/60 flex items-center justify-between">
            <span className="text-sm text-gray-500">v1 legacy endpoints</span>
            <span className="text-xs font-semibold text-yellow-400 bg-yellow-950/40 border border-yellow-800/50 px-2 py-0.5 rounded uppercase">Deprecated</span>
          </div>
          <div className="py-3 flex gap-2">
            <Info size={13} className="text-gray-600 shrink-0 mt-0.5" />
            <p className="text-xs text-gray-600">
              Legacy v1 routes (<code className="font-mono text-gray-500">/predict/failure</code>,{" "}
              <code className="font-mono text-gray-500">/work-orders</code>, etc.) respond with{" "}
              <code className="font-mono text-gray-500">Deprecation: true</code> and{" "}
              <code className="font-mono text-gray-500">Sunset</code> headers.
              They will be removed in v3.
            </p>
          </div>
        </div>
      </section>

      {/* ── ADR ──────────────────────────────────────────────────────────── */}
      <section>
        <SectionHeader title="Architecture Decision Records" subtitle="Key design choices" />
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5">
          <AdrNote
            code="ADR-01"
            title="No real authentication"
            decision="Demo scope — all endpoints are public. Production would add JWT/OAuth2 via FastAPI-Security."
          />
          <AdrNote
            code="ADR-02"
            title="Mock CMMS integration"
            decision="Work orders are stored locally in PostgreSQL. A production deployment would sync with SAP PM / IBM Maximo via REST webhook."
          />
          <AdrNote
            code="ADR-03"
            title="XGBoost over deep learning"
            decision="Only ~48 distinct failure events in training set. LSTM/Transformer would overfit; XGBoost gives stable SHAP explanations and sub-second inference."
          />
          <AdrNote
            code="ADR-04"
            title="SQLite MLflow tracking"
            decision="No MLflow server running in demo. MLFLOW_TRACKING_URI points to a local SQLite file. Production: managed MLflow on PostgreSQL + S3 artifacts."
          />
          <AdrNote
            code="ADR-05"
            title="Structured logging (structlog)"
            decision="All backend logs emit JSON lines. Set LOG_FILE_PATH env var to stream them into the Logs Explorer page."
          />
        </div>
      </section>
    </div>
  );
}
