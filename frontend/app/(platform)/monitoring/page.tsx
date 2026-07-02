"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmtTs } from "@/lib/utils";
import { CheckCircle2, XCircle, Clock } from "lucide-react";

function ServiceCard({ name, status, detail }: { name: string; status: string; detail?: string }) {
  const isOk = status === "ok";
  const isTimeout = status === "timeout";
  const Icon = isOk ? CheckCircle2 : isTimeout ? Clock : XCircle;
  const iconColor = isOk ? "text-green-400" : isTimeout ? "text-yellow-400" : "text-red-400";
  const borderColor = isOk ? "border-green-800/40" : isTimeout ? "border-yellow-800/40" : "border-red-800/40";
  const bgColor = isOk ? "bg-green-950/20" : isTimeout ? "bg-yellow-950/20" : "bg-red-950/20";

  return (
    <div className={`border rounded-xl p-5 flex flex-col gap-3 ${borderColor} ${bgColor}`}>
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-gray-200 capitalize">{name}</p>
        <Icon size={16} className={iconColor} />
      </div>
      <p className={`text-xs font-medium uppercase tracking-wider ${iconColor}`}>{status}</p>
      {detail && <p className="text-xs text-gray-500">{detail}</p>}
    </div>
  );
}

export default function MonitoringPage() {
  const { data, isLoading, dataUpdatedAt } = useQuery({
    queryKey: ["monitoring-services"],
    queryFn: api.monitoring.services,
    refetchInterval: 30_000,
  });

  return (
    <div className="px-8 py-8 max-w-4xl">
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Services</h1>
          <p className="text-sm text-gray-500 mt-0.5">Backend infrastructure health</p>
        </div>
        <div className="flex items-center gap-3">
          {data && (
            <span className={`text-sm font-semibold uppercase ${data.overall === "ok" ? "text-green-400" : "text-yellow-400"}`}>
              {data.overall}
            </span>
          )}
          {dataUpdatedAt > 0 && (
            <p className="text-xs text-gray-600">
              Checked {fmtTs(new Date(dataUpdatedAt).toISOString())}
            </p>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 animate-pulse">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-gray-800/40" />
          ))}
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {data.services.map((s) => (
            <ServiceCard key={s.name} {...s} />
          ))}
        </div>
      )}
    </div>
  );
}
