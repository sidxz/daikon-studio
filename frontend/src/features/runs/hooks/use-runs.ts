"use client";

import {
  API_V1,
  customInstance,
  getApiBaseUrl,
  getAuthHeaders,
} from "@/shared/lib/api/custom-instance";
import type {
  PaginatedResponsePredictionResponse,
  PaginatedResponseRunResponse,
  PredictBody,
  RunResponse,
  UploadResponse,
} from "@/shared/lib/api/model";
import { pollInterval } from "@/shared/lib/query-defaults";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ResultParams } from "../lib/result-query";
import type { TriageRow } from "../types";
import { RUNS_KEY, RUN_KEY } from "./query-keys";

/** Prediction runs, newest first. Training runs live on their Protocol. */
export function useRuns(
  kind: "prediction" | "training" | undefined = "prediction",
  cursor?: string,
) {
  return useQuery({
    queryKey: [...RUNS_KEY, kind ?? null, cursor ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseRunResponse>({
        url: `${API_V1}/runs`,
        method: "GET",
        params: { kind, cursor },
      }),
  });
}

export function useRun(id: string | undefined) {
  return useQuery({
    queryKey: [...RUN_KEY, id],
    queryFn: () => customInstance<RunResponse>({ url: `${API_V1}/runs/${id}`, method: "GET" }),
    enabled: Boolean(id),
    refetchInterval: pollInterval,
  });
}

export function useUploadPredictionFile() {
  return useMutation({
    mutationFn: async (file: File): Promise<string> => {
      const body = new FormData();
      body.append("file", file);
      // Bare fetch: the browser must set its own multipart boundary.
      const response = await fetch(`${getApiBaseUrl()}${API_V1}/datasets/uploads`, {
        method: "POST",
        headers: getAuthHeaders(),
        body,
      });
      if (!response.ok) throw new Error(`Upload failed (${response.status})`);
      return ((await response.json()) as UploadResponse).upload_ref;
    },
  });
}

export function useCreateRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: PredictBody) =>
      customInstance<RunResponse>({ url: `${API_V1}/runs`, method: "POST", data }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: RUNS_KEY }),
  });
}

export function useCancelRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/runs/${id}/cancel`, method: "POST" }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: [...RUN_KEY, id] });
      showSuccess("Run canceled");
    },
  });
}

/** Failed or cancelled back to pending, same Run, same id; the server refuses anything else. */
export function useRetryRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/runs/${id}/retry`, method: "POST" }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: [...RUN_KEY, id] });
      showSuccess("Run requeued");
    },
  });
}

/**
 * One page of a run's results, each row carrying the server's `row_id`.
 *
 * Not a hook: AG Grid's infinite row model calls its datasource imperatively
 * with a start row, so this is a plain async function. The cursor for this
 * endpoint is a plain integer offset -- unlike the base64 keyset cursors the
 * other lists use, which is a real trap given both surface as `next_cursor`.
 */
export async function fetchResultBlock(
  runId: string,
  startRow: number,
  limit: number,
  params: ResultParams,
): Promise<{ rows: TriageRow[]; nextCursor: string | null }> {
  const page = await customInstance<PaginatedResponsePredictionResponse>({
    url: `${API_V1}/runs/${runId}/results`,
    method: "GET",
    params: { cursor: String(startRow), limit, ...params },
  });
  return {
    // `row_id` from the server, never the page offset: under a sort or filter
    // the two disagree, and the offset would save the wrong compounds into a
    // Collection without any visible symptom.
    rows: page.items.map((item) => ({ ...item, __rowId: item.row_id })),
    nextCursor: page.next_cursor ?? null,
  };
}
