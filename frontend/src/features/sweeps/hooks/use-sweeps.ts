"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type { SweepDetailResponse, SweepListResponse } from "@/shared/lib/api/model";
import { RUN_POLL_MS, isTerminal } from "@/shared/lib/query-defaults";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { SWEEPS_KEY, SWEEP_KEY } from "./query-keys";

export function useSweeps() {
  return useQuery({
    queryKey: SWEEPS_KEY,
    queryFn: () => customInstance<SweepListResponse>({ url: `${API_V1}/sweeps`, method: "GET" }),
  });
}

/**
 * Polls until every member is terminal. A sweep is throttled by the
 * per-workspace concurrency cap, so the tail of a large one keeps arriving
 * long after the first few finish -- stopping at the first terminal run would
 * freeze the page mid-comparison. A failed request stops it, as `pollInterval`
 * does for one Run; a sweep has no single status for that helper to read.
 */
export function useSweep(id: string | undefined) {
  return useQuery({
    queryKey: [...SWEEP_KEY, id],
    queryFn: () =>
      customInstance<SweepDetailResponse>({ url: `${API_V1}/sweeps/${id}`, method: "GET" }),
    enabled: Boolean(id),
    refetchInterval: (query) => {
      if (query.state.status === "error") return false;
      const runs = query.state.data?.runs ?? [];
      return runs.length > 0 && runs.every((run) => isTerminal(run.status)) ? false : RUN_POLL_MS;
    },
  });
}

export function useSubmitSweep() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      dataset_id: string;
      configs: { engine_id: string; conditions: Record<string, unknown> }[];
      baseline_engine_id?: string | null;
      baseline_conditions?: Record<string, unknown>;
      tune_cutoffs?: boolean;
    }) => customInstance<SweepDetailResponse>({ url: `${API_V1}/sweeps`, method: "POST", data }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: SWEEPS_KEY }),
  });
}

export function useCancelSweep() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/sweeps/${id}/cancel`, method: "POST" }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: SWEEPS_KEY });
      queryClient.invalidateQueries({ queryKey: [...SWEEP_KEY, id] });
    },
  });
}
