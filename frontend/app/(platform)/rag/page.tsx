"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, type RagQuery } from "@/lib/api";
import { fmtTs, fmtDuration } from "@/lib/utils";
import { Search, CheckCircle2, XCircle, Database } from "lucide-react";

function StatTile({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-5 py-4">
      <p className="text-[11px] text-gray-600 uppercase tracking-wider">{label}</p>
      <p className={`text-3xl font-bold tabular-nums mt-1 leading-none ${accent ?? "text-gray-100"}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-1">{sub}</p>}
    </div>
  );
}

function HitBadge({ hit }: { hit: boolean }) {
  return hit
    ? <span className="flex items-center gap-1 text-[11px] text-green-400"><CheckCircle2 size={11} /> hit</span>
    : <span className="flex items-center gap-1 text-[11px] text-red-400"><XCircle size={11} /> miss</span>;
}

export default function RagPage() {
  const [hitFilter, setHitFilter] = useState<string>("");
  const [search, setSearch] = useState("");

  const { data: summary } = useQuery({
    queryKey: ["rag-summary"],
    queryFn: api.rag.summary,
    refetchInterval: 30_000,
  });

  const { data: queries, isLoading } = useQuery({
    queryKey: ["rag-queries", hitFilter],
    queryFn: () =>
      api.rag.queries({
        limit: 100,
        hit: hitFilter === "hit" ? true : hitFilter === "miss" ? false : undefined,
      }),
    refetchInterval: 20_000,
  });

  const filtered = (queries?.queries ?? []).filter(
    (q) => !search || q.query.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="px-8 py-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-gray-100">RAG Dashboard</h1>
        <p className="text-sm text-gray-500 mt-0.5">Retrieval-Augmented Generation performance &amp; query log</p>
      </div>

      {/* Summary tiles */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <StatTile
            label="Total Queries"
            value={String(summary.total_queries)}
            sub={`${summary.hits} hits / ${summary.misses} misses`}
          />
          <StatTile
            label="Miss Rate"
            value={`${(summary.miss_rate * 100).toFixed(1)}%`}
            accent={summary.miss_rate > 0.2 ? "text-red-400" : summary.miss_rate > 0.1 ? "text-yellow-400" : "text-green-400"}
          />
          <StatTile
            label="Avg Latency"
            value={fmtDuration(summary.avg_latency_ms)}
            sub="retrieval only"
            accent={summary.avg_latency_ms > 500 ? "text-yellow-400" : "text-gray-100"}
          />
          <StatTile
            label="Avg Similarity"
            value={summary.avg_similarity.toFixed(3)}
            sub={summary.chunk_count != null ? `${summary.chunk_count.toLocaleString()} chunks indexed` : undefined}
            accent={summary.avg_similarity >= 0.7 ? "text-green-400" : "text-yellow-400"}
          />
        </div>
      )}

      {/* Hit/miss ratio bar */}
      {summary && summary.total_queries > 0 && (
        <div className="mb-8 bg-gray-900/60 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Hit / Miss ratio</p>
            <div className="flex items-center gap-4 text-xs">
              <span className="text-green-400">● {summary.hits} hits</span>
              <span className="text-red-400">● {summary.misses} misses</span>
            </div>
          </div>
          <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
            <div
              className="bg-green-500 transition-all"
              style={{ width: `${(summary.hits / summary.total_queries) * 100}%` }}
            />
            <div
              className="bg-red-500 transition-all"
              style={{ width: `${(summary.misses / summary.total_queries) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Query log */}
      <div>
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Query Log</h2>

        <div className="flex gap-3 mb-4 flex-wrap">
          <div className="flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm flex-1 min-w-48">
            <Search size={13} className="text-gray-500 shrink-0" />
            <input
              type="text"
              placeholder="Search queries…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-transparent outline-none text-gray-200 placeholder-gray-600 w-full text-sm"
            />
          </div>
          <select
            value={hitFilter}
            onChange={(e) => setHitFilter(e.target.value)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 outline-none cursor-pointer"
          >
            <option value="">All results</option>
            <option value="hit">Hits only</option>
            <option value="miss">Misses only</option>
          </select>
          <span className="text-xs text-gray-600 self-center">{filtered.length} queries</span>
        </div>

        <div className="border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900/80 border-b border-gray-800">
              <tr className="text-[11px] text-gray-500 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Query</th>
                <th className="px-4 py-3 text-center">Result</th>
                <th className="px-4 py-3 text-right">Similarity</th>
                <th className="px-4 py-3 text-right">Latency</th>
                <th className="px-4 py-3 text-right">Top-K</th>
                <th className="px-4 py-3 text-left">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60">
              {isLoading && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-600 text-xs">Loading…</td></tr>
              )}
              {filtered.map((q) => (
                <tr key={q.id} className="hover:bg-gray-900/40 transition-colors">
                  <td className="px-4 py-3 text-gray-300 text-xs max-w-xs">
                    <p className="truncate">{q.query}</p>
                    {q.sources && q.sources.length > 0 && (
                      <p className="text-gray-600 text-[11px] mt-0.5 truncate">
                        {q.sources.slice(0, 2).join(", ")}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-center"><HitBadge hit={q.hit} /></td>
                  <td className="px-4 py-3 text-right tabular-nums text-xs text-gray-400">
                    {q.avg_similarity?.toFixed(3) ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-xs text-gray-400">
                    {fmtDuration(q.retrieval_latency_ms)}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-xs text-gray-500">{q.top_k}</td>
                  <td className="px-4 py-3 text-xs text-gray-600 whitespace-nowrap">{fmtTs(q.created_at)}</td>
                </tr>
              ))}
              {!isLoading && filtered.length === 0 && (
                <tr><td colSpan={6} className="px-4 py-10 text-center text-gray-600 text-sm">No queries match.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
