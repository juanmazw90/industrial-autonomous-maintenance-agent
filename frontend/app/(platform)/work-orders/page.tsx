"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type WorkOrder, type WorkOrderStatus, type WorkOrderPriority } from "@/lib/api";
import { fmt, fmtTs } from "@/lib/utils";
import { Plus, X, AlertCircle, ChevronRight, Wrench } from "lucide-react";

// ── Priority config ──────────────────────────────────────────────────────────

const PRIORITY_CFG: Record<WorkOrderPriority, { label: string; cls: string }> = {
  critical: { label: "Crítico", cls: "text-red-400 bg-red-950/40 border-red-800/50" },
  high:     { label: "Alto",    cls: "text-orange-400 bg-orange-950/40 border-orange-800/50" },
  medium:   { label: "Medio",   cls: "text-yellow-400 bg-yellow-950/40 border-yellow-800/50" },
  low:      { label: "Bajo",    cls: "text-gray-400 bg-gray-800/40 border-gray-700/50" },
};

const COLUMNS: { status: WorkOrderStatus; label: string; accent: string }[] = [
  { status: "open",        label: "Abiertas",   accent: "border-gray-700" },
  { status: "assigned",    label: "Asignadas",  accent: "border-blue-800/60" },
  { status: "in_progress", label: "En Progreso", accent: "border-yellow-800/60" },
  { status: "completed",   label: "Completadas", accent: "border-green-800/60" },
];

const NEXT_STATUS_LABEL: Record<WorkOrderStatus, string> = {
  open:        "Asignada",
  assigned:    "En Progreso",
  in_progress: "Completada",
  completed:   "",
};

const NEXT_STATUS: Record<WorkOrderStatus, WorkOrderStatus | null> = {
  open:        "assigned",
  assigned:    "in_progress",
  in_progress: "completed",
  completed:   null,
};

function PriorityBadge({ priority }: { priority: string }) {
  const cfg = PRIORITY_CFG[priority as WorkOrderPriority] ?? PRIORITY_CFG.medium;
  return (
    <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

// ── Work Order card ───────────────────────────────────────────────────────────

function WOCard({ wo, onAdvance }: { wo: WorkOrder; onAdvance: (id: string, next: WorkOrderStatus) => void }) {
  const next = NEXT_STATUS[wo.status];
  const nextLabel = next ? NEXT_STATUS_LABEL[wo.status] : null;

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 space-y-3 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-medium text-gray-200 leading-snug">{wo.title}</p>
        <PriorityBadge priority={wo.priority} />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] font-mono text-gray-500 bg-gray-800/50 px-2 py-0.5 rounded">
          {wo.machine_code}
        </span>
        {wo.estimated_cost != null && (
          <span className="text-[11px] text-gray-600">{fmt(wo.estimated_cost, "currency")}</span>
        )}
        {wo.assigned_to && (
          <span className="text-[11px] text-gray-600 truncate max-w-[8rem]">→ {wo.assigned_to}</span>
        )}
      </div>

      {wo.description && (
        <p className="text-xs text-gray-600 line-clamp-2">{wo.description}</p>
      )}

      <div className="flex items-center justify-between pt-1">
        <p className="text-[11px] text-gray-700">{fmtTs(wo.created_at)}</p>
        {nextLabel && (
          <button
            onClick={() => onAdvance(wo.id, next!)}
            className="flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 border border-indigo-900/60 hover:border-indigo-700/60 px-2 py-1 rounded-lg transition-colors"
          >
            Mover a {nextLabel}
            <ChevronRight size={11} />
          </button>
        )}
        {wo.status === "completed" && (
          <span className="text-[11px] text-green-500 font-medium">Completado</span>
        )}
      </div>
    </div>
  );
}

// ── Column ────────────────────────────────────────────────────────────────────

function KanbanColumn({
  status, label, accent, orders, onAdvance,
}: {
  status: WorkOrderStatus;
  label: string;
  accent: string;
  orders: WorkOrder[];
  onAdvance: (id: string, next: WorkOrderStatus) => void;
}) {
  return (
    <div className="flex flex-col min-w-[260px] flex-1">
      <div className={`flex items-center justify-between border-b-2 ${accent} pb-2 mb-3`}>
        <h3 className="text-sm font-semibold text-gray-300">{label}</h3>
        <span className="text-xs text-gray-600 bg-gray-800/60 px-2 py-0.5 rounded-full">{orders.length}</span>
      </div>
      <div className="flex flex-col gap-3">
        {orders.length === 0 && (
          <div className="py-8 text-center text-gray-700 text-xs border border-dashed border-gray-800 rounded-xl">
            Sin órdenes de trabajo
          </div>
        )}
        {orders.map((wo) => (
          <WOCard key={wo.id} wo={wo} onAdvance={onAdvance} />
        ))}
      </div>
    </div>
  );
}

// ── Create modal ──────────────────────────────────────────────────────────────

function CreateModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    machine_code: "",
    title: "",
    description: "",
    priority: "medium",
    estimated_cost: "",
  });
  const [err, setErr] = useState("");

  const mutation = useMutation({
    mutationFn: api.workOrders.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["work-orders"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof Error ? e.message : "Error al crear la orden de trabajo"),
  });

  function submit() {
    if (!form.machine_code.trim()) { setErr("El código de máquina es obligatorio"); return; }
    if (!form.title.trim()) { setErr("El título es obligatorio"); return; }
    const cost = form.estimated_cost ? parseFloat(form.estimated_cost) : undefined;
    mutation.mutate({
      machine_code: form.machine_code.trim().toUpperCase(),
      title: form.title.trim(),
      description: form.description.trim() || undefined,
      priority: form.priority,
      estimated_cost: isNaN(cost as number) ? undefined : cost,
    });
  }

  const inp = "bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-indigo-700 w-full";

  return (
    <div className="fixed inset-0 bg-gray-950/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 w-full max-w-lg space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Nueva Orden de Trabajo</h3>
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300"><X size={16} /></button>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Código de Máquina *</label>
              <input className={inp} placeholder="PUMP-001" value={form.machine_code}
                onChange={(e) => setForm({ ...form, machine_code: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Prioridad</label>
              <select className={inp + " cursor-pointer"} value={form.priority}
                onChange={(e) => setForm({ ...form, priority: e.target.value })}>
                <option value="critical">Crítico</option>
                <option value="high">Alto</option>
                <option value="medium">Medio</option>
                <option value="low">Bajo</option>
              </select>
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Título *</label>
            <input className={inp} placeholder="Inspección y lubricación de rodamientos" value={form.title}
              onChange={(e) => setForm({ ...form, title: e.target.value })} />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Descripción</label>
            <textarea className={inp + " resize-none"} rows={2} placeholder="Detalles opcionales…"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })} />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Costo Estimado (USD)</label>
            <input className={inp} type="number" min={0} step={0.01} placeholder="0.00"
              value={form.estimated_cost}
              onChange={(e) => setForm({ ...form, estimated_cost: e.target.value })} />
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
            {mutation.isPending ? "Creando…" : "Crear Orden de Trabajo"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function WorkOrdersPage() {
  const [showCreate, setShowCreate] = useState(false);
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["work-orders"],
    queryFn: () => api.workOrders.list({ limit: 200 }),
    refetchInterval: 20_000,
  });

  const advanceMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: WorkOrderStatus }) =>
      api.workOrders.update(id, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["work-orders"] }),
  });

  const allOrders = data?.work_orders ?? [];

  const byStatus = (status: WorkOrderStatus) =>
    allOrders.filter((wo) => wo.status === status);

  function handleAdvance(id: string, next: WorkOrderStatus) {
    advanceMutation.mutate({ id, status: next });
  }

  return (
    <div className="px-8 py-8 h-full flex flex-col">
      <div className="flex items-baseline justify-between mb-8 shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Órdenes de Trabajo</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {data?.total ?? 0} en total — seguimiento de tareas de mantenimiento
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Summary pills */}
          {!isLoading && (
            <div className="flex gap-2">
              {COLUMNS.slice(0, 3).map(({ status, label }) => {
                const count = byStatus(status).length;
                if (count === 0) return null;
                return (
                  <span key={status} className="text-xs text-gray-500 bg-gray-900 border border-gray-800 px-2.5 py-1 rounded-full">
                    {count} {label.toLowerCase()}
                  </span>
                );
              })}
            </div>
          )}
          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-2 text-sm bg-indigo-700 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <Plus size={14} />Nueva
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="flex gap-4 flex-1">
          {COLUMNS.map(({ status }) => (
            <div key={status} className="flex-1 space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-28 rounded-xl bg-gray-800/40 animate-pulse" />
              ))}
            </div>
          ))}
        </div>
      )}

      {!isLoading && (
        <div className="flex gap-4 flex-1 overflow-x-auto pb-2">
          {COLUMNS.map(({ status, label, accent }) => (
            <KanbanColumn
              key={status}
              status={status}
              label={label}
              accent={accent}
              orders={byStatus(status)}
              onAdvance={handleAdvance}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && allOrders.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
          <Wrench size={32} className="text-gray-700" />
          <p className="text-sm text-gray-500">Sin órdenes de trabajo aún.</p>
          <p className="text-xs text-gray-700">Créala manualmente o deja que el agente IA las genere desde las alertas.</p>
        </div>
      )}

      {showCreate && <CreateModal onClose={() => setShowCreate(false)} />}
    </div>
  );
}
