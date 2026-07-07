/**
 * hooks.ts — Data hooks compartidos (React Query).
 *
 * Centraliza queries usadas por varias páginas para que compartan queryKey
 * (deduplicación) y una única definición de fetcher/intervalo.
 */

import { useQuery } from "@tanstack/react-query";
import { api, type OperationsSummary } from "@/lib/api";

export function useOperationsSummary(refetchInterval: number = 15_000) {
  return useQuery<OperationsSummary>({
    queryKey: ["operations-summary"],
    queryFn: api.operations.summary,
    refetchInterval,
  });
}
