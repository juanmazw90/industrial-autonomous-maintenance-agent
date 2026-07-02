"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type Evaluation } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { ClipboardCheck, Plus, X, AlertCircle } from "lucide-react";

function ScoreBadge({ value, inverted = false }: { value: number | null; inverted?: boolean }) {
  if (value == null) return <span className="text-gray-600">—</span>;
  const pct = Math.round(value * 100);
  const good = inverted ? pct < 20 : pct >= 80;
  const warn = inverted ? pct < 50 : pct >= 50;
  const color = good ? "text-green-400" : warn ? "text-yellow-400" : "text-red-400";
  return <span className={`tabular-nums font-semibold ${color}`}>{pct}%</span>;
}

// ── KPI summary from stored evaluations ──────────────────────────────────────

function KpiRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-800/60 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-semibold">{value}</span>
    </div>
  );
}

function SummaryPanel({ evals }: { evals: Evaluation[] }) {
  if (evals.length === 0) return null;
  const avg = (key: keyof Evaluation) => {
    const vals = evals.map((e) => e[key] as number | null).filter((v) => v != null) as number[];
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const hallucinationRate = evals.filter((e) => e.hallucination_flag).length / evals.length;

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 mb-8">
      <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-2">KPIs Agregados ({evals.length} evaluaciones)</h2>
      <KpiRow label="Precisión Media"       value={<ScoreBadge value={avg("accuracy")} />} />
      <KpiRow label="Fundamentación Media"  value={<ScoreBadge value={avg("groundedness")} />} />
      <KpiRow label="Tasa de Alucinación"  value={<ScoreBadge value={hallucinationRate} inverted />} />
      <KpiRow label="Éxito de Herramientas" value={<ScoreBadge value={avg("tool_success_rate")} />} />
    </div>
  );
}

// ── Create evaluation form ────────────────────────────────────────────────────

function CreateForm({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    agent_run_id: "",
    accuracy: "",
    groundedness: "",
    hallucination_flag: false,
    tool_success_rate: "",
    evaluator: "human",
    notes: "",
  });
  const [err, setErr] = useState("");

  const mutation = useMutation({
    mutationFn: api.evaluations.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["evaluations"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof Error ? e.message : "Error"),
  });

  function parseOptFloat(s: string): number | undefined {
    const n = parseFloat(s);
    return isNaN(n) ? undefined : Math.max(0, Math.min(1, n));
  }

  function submit() {
    if (!form.agent_run_id.trim()) { setErr("El ID de ejecución es obligatorio"); return; }
    mutation.mutate({
      agent_run_id: form.agent_run_id.trim(),
      accuracy: parseOptFloat(form.accuracy),
      groundedness: parseOptFloat(form.groundedness),
      hallucination_flag: form.hallucination_flag,
      tool_success_rate: parseOptFloat(form.tool_success_rate),
      evaluator: form.evaluator || undefined,
      notes: form.notes || undefined,
    });
  }

  const inp = "bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-indigo-700 w-full";

  return (
    <div className="fixed inset-0 bg-gray-950/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 w-full max-w-lg space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Nueva Evaluación</h3>
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300"><X size={16} /></button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-gray-500 mb-1 block">ID de Ejecución *</label>
            <input className={inp} placeholder="uuid…" value={form.agent_run_id}
              onChange={(e) => setForm({ ...form, agent_run_id: e.target.value })} />
          </div>
          <div className="grid grid-cols-3 gap-3">
            {(["accuracy", "groundedness", "tool_success_rate"] as const).map((k) => (
              <div key={k}>
                <label className="text-xs text-gray-500 mb-1 block capitalize">{k.replace(/_/g, " ")} (0–1)</label>
                <input className={inp} type="number" min={0} max={1} step={0.01} placeholder="0.00–1.00"
                  value={(form as Record<string, string | boolean>)[k] as string}
                  onChange={(e) => setForm({ ...form, [k]: e.target.value })} />
              </div>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <input type="checkbox" id="hal" checked={form.hallucination_flag}
              onChange={(e) => setForm({ ...form, hallucination_flag: e.target.checked })}
              className="accent-red-500" />
            <label htmlFor="hal" className="text-sm text-gray-400">Alucinación detectada</label>
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Evaluador</label>
            <input className={inp} placeholder="human / llm-judge / batch" value={form.evaluator}
              onChange={(e) => setForm({ ...form, evaluator: e.target.value })} />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Notas</label>
            <textarea className={inp + " resize-none"} rows={2} value={form.notes}
              onChange={(e) => setForm({ ...form, notes: e.target.value })} />
          </div>
        </div>

        {err && (
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle size={12} />{err}
          </div>
        )}

        <div className="flex gap-3 pt-1">
          <button onClick={onClose}
            className="flex-1 py-2 text-sm border border-gray-700 rounded-lg text-gray-400 hover:text-gray-200 transition-colors">
            Cancelar
          </button>
          <button onClick={submit} disabled={mutation.isPending}
            className="flex-1 py-2 text-sm bg-indigo-700 hover:bg-indigo-600 text-white rounded-lg transition-colors disabled:opacity-50">
            {mutation.isPending ? "Guardando…" : "Guardar Evaluación"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function EvaluationPage() {
  const [showForm, setShowForm] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["evaluations"],
    queryFn: () => api.evaluations.list({ limit: 100 }),
    refetchInterval: 30_000,
  });

  const evals = data?.evaluations ?? [];

  return (
    <div className="px-8 py-8 max-w-5xl">
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Evaluación de Agentes</h1>
          <p className="text-sm text-gray-500 mt-0.5">Calidad de respuesta, fundamentación y seguimiento de alucinaciones</p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="flex items-center gap-2 text-sm bg-indigo-700 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <Plus size={14} />Nueva Evaluación
        </button>
      </div>

      <SummaryPanel evals={evals} />

      {/* Evaluations table */}
      <div className="border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-900/80 border-b border-gray-800">
            <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-3 text-left">ID de Ejecución</th>
              <th className="px-4 py-3 text-center">Precisión</th>
              <th className="px-4 py-3 text-center">Fundamentación</th>
              <th className="px-4 py-3 text-center">Alucinación</th>
              <th className="px-4 py-3 text-center">Éxito Tool</th>
              <th className="px-4 py-3 text-left">Evaluador</th>
              <th className="px-4 py-3 text-left">Creado</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {isLoading && (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-xs">Cargando…</td></tr>
            )}
            {evals.map((e) => (
              <tr key={e.id} className="hover:bg-gray-900/40 transition-colors group">
                <td className="px-4 py-3">
                  <span className="text-xs font-mono text-gray-500">{e.agent_run_id.slice(0, 12)}…</span>
                  {e.notes && <p className="text-[11px] text-gray-600 mt-0.5 line-clamp-1">{e.notes}</p>}
                </td>
                <td className="px-4 py-3 text-center"><ScoreBadge value={e.accuracy} /></td>
                <td className="px-4 py-3 text-center"><ScoreBadge value={e.groundedness} /></td>
                <td className="px-4 py-3 text-center">
                  {e.hallucination_flag
                    ? <span className="text-red-400 text-xs font-semibold">YES</span>
                    : <span className="text-green-400 text-xs">no</span>}
                </td>
                <td className="px-4 py-3 text-center"><ScoreBadge value={e.tool_success_rate} /></td>
                <td className="px-4 py-3 text-xs text-gray-500">{e.evaluator ?? "—"}</td>
                <td className="px-4 py-3 text-xs text-gray-600">{fmtTs(e.created_at)}</td>
              </tr>
            ))}
            {!isLoading && evals.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-16 text-center space-y-2">
                  <ClipboardCheck size={28} className="mx-auto text-gray-700" />
                  <p className="text-sm text-gray-600">Sin evaluaciones aún.</p>
                  <p className="text-xs text-gray-700">Haz clic en "Nueva Evaluación" para registrar una revisión manual.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {showForm && <CreateForm onClose={() => setShowForm(false)} />}
    </div>
  );
}
