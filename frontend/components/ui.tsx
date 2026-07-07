/**
 * ui.tsx — Primitivas de UI compartidas entre páginas de la plataforma.
 *
 * Antes cada página redeclaraba sus propios Tile/KpiCard/SectionTitle/Skeleton.
 * Cualquier ajuste visual se hace aquí una sola vez.
 */

import { cn } from "@/lib/utils";

/** Contenedor estándar de sección/panel. */
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn("bg-gray-900/60 border border-gray-800 rounded-xl", className)}>
      {children}
    </div>
  );
}

/** Título de sección con ícono (uppercase, gris). */
export function SectionTitle({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon size={14} className="text-gray-500" />
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{title}</h2>
    </div>
  );
}

/** Tile KPI grande (dashboard). */
export function Tile({
  label,
  value,
  sub,
  color = "text-gray-100",
  icon: Icon,
  pulse,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  icon?: React.ElementType;
  pulse?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5 bg-gray-900/60 border border-gray-800 rounded-xl px-5 py-4 min-w-0">
      <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wider">
        {Icon && <Icon size={12} className="shrink-0" />}
        {label}
      </div>
      <p className={cn("text-3xl font-bold tabular-nums leading-none", color, pulse && "animate-pulse")}>
        {value}
      </p>
      {sub && <p className="text-xs text-gray-600">{sub}</p>}
    </div>
  );
}

/** Tarjeta KPI compacta (monitoring). */
export function KpiCard({
  label,
  value,
  sub,
  accent = "text-gray-100",
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3">
      <p className="text-[11px] text-gray-600 uppercase tracking-wider">{label}</p>
      <p className={cn("text-2xl font-bold tabular-nums mt-0.5 leading-none", accent)}>{value}</p>
      {sub && <p className="text-[11px] text-gray-600 mt-1">{sub}</p>}
    </div>
  );
}

/** Punto de estado ok/fallo/desconocido. */
export function StatusDot({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="w-2 h-2 rounded-full bg-gray-600 inline-block" />;
  return ok
    ? <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
    : <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse inline-block" />;
}

/** Bloque skeleton con animación de pulso. */
export function SkeletonBlock({ className }: { className?: string }) {
  return <div className={cn("rounded-xl bg-gray-800/50 animate-pulse", className)} />;
}
