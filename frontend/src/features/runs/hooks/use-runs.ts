"use client";

import {
  API_V1,
  customInstance,
  getApiBaseUrl,
  getAuthHeaders,
} from "@/shared/lib/api/custom-instance";
import { downloadFile } from "@/shared/lib/api/download";
import type {
  ColumnRangeResponse,
  EpochResponse,
  PaginatedResponsePredictionResponse,
  PaginatedResponseRunResponse,
  PredictBody,
  RunChemicalSpaceResponse,
  RunMapCompoundResponse,
  RunResponse,
  UploadResponse,
} from "@/shared/lib/api/model";
import { STALE_TIME, mapStaleTime, pollInterval } from "@/shared/lib/query-defaults";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ResultParams } from "../lib/result-query";
import type { TriageRow } from "../types";
import { RUNS_KEY, RUN_KEY } from "./query-keys";

/** Prediction runs, newest first. Training runs live on their Protocol. */
/** A run's compounds placed on its protocol's map. Fixed once the run is ready. */
export function useRunChemicalSpace(id: string | undefined) {
  return useQuery({
    queryKey: [...RUN_KEY, id, "chemical-space"],
    queryFn: () =>
      customInstance<RunChemicalSpaceResponse>({
        url: `${API_V1}/runs/${id}/chemical-space`,
        method: "GET",
      }),
    enabled: Boolean(id),
    staleTime: mapStaleTime,
  });
}

/**
 * Each numeric results column's range across the whole run: the scale the triage
 * grid's in-cell bars are drawn against. A finished run's results never change.
 */
export function useResultRanges(id: string) {
  return useQuery({
    queryKey: [...RUN_KEY, id, "results", "ranges"],
    queryFn: () =>
      customInstance<Record<string, ColumnRangeResponse>>({
        url: `${API_V1}/runs/${id}/results/ranges`,
        method: "GET",
      }),
    staleTime: STALE_TIME.LONG,
  });
}

/** One scored compound, for a hover tooltip. Null row means nothing hovered. */
export function useRunMapCompound(id: string, row: number | null) {
  return useQuery({
    queryKey: [...RUN_KEY, id, "chemical-space", "compound", row],
    queryFn: async () =>
      (
        await customInstance<RunMapCompoundResponse[]>({
          url: `${API_V1}/runs/${id}/chemical-space/compounds`,
          method: "GET",
          params: { rows: [row] },
        })
      )[0] ?? null,
    enabled: row !== null && row >= 0,
    staleTime: STALE_TIME.LONG,
  });
}

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

const EPOCH_POLL_MS = 5000;

/**
 * A training run's finished epochs, for its live charts: polled while it trains, read
 * once when it has finished. The runner sends them in batches every ten seconds or so,
 * so polling faster would only redraw the same points.
 */
export function useRunEpochs(id: string, live: boolean) {
  return useQuery({
    queryKey: [...RUN_KEY, id, "epochs"],
    queryFn: () =>
      customInstance<EpochResponse[]>({ url: `${API_V1}/runs/${id}/epochs`, method: "GET" }),
    enabled: Boolean(id),
    refetchInterval: live ? EPOCH_POLL_MS : false,
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

/**
 * Failed or cancelled back to pending, same Run, same id; the server refuses anything else.
 * A training run resumes from its saved progress; `fresh` discards that and starts over.
 */
export function useRetryRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, fresh = false }: { id: string; fresh?: boolean }) =>
      customInstance<void>({
        url: `${API_V1}/runs/${id}/retry`,
        method: "POST",
        ...(fresh ? { data: { fresh: true } } : {}),
      }),
    onSuccess: (_data, { id, fresh }) => {
      queryClient.invalidateQueries({ queryKey: [...RUN_KEY, id] });
      queryClient.invalidateQueries({ queryKey: RUNS_KEY });
      showSuccess(fresh ? "Starting over" : "Run requeued");
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

/** The results as a workbook, under the grid's own sort and filters. Through fetch,
 * not a link: the token travels as a header. Failures toast like any mutation's. */
export function useExportRunResults() {
  return useMutation({
    mutationFn: ({
      runId,
      params,
      filename,
    }: {
      runId: string;
      params: ResultParams;
      filename: string;
    }) => {
      const query = new URLSearchParams(
        Object.entries(params).filter((entry): entry is [string, string] => entry[1] != null),
      ).toString();
      return downloadFile({
        url: `${API_V1}/runs/${runId}/results/export${query ? `?${query}` : ""}`,
        filename,
      });
    },
  });
}
