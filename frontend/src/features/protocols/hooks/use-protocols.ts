"use client";

import { API_V1, ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import type {
  ChemicalSpaceResponse,
  MapCompoundResponse,
  PaginatedResponseProtocolResponse,
  PaginatedResponseRunResponse,
  ProtocolResponse,
  RunResponse,
  ScorecardResponse,
  TrainProtocolBody,
} from "@/shared/lib/api/model";
import { STALE_TIME, mapStaleTime, pollInterval } from "@/shared/lib/query-defaults";
import { showError, showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PROTOCOLS_KEY, PROTOCOL_KEY, PROTOCOL_RUNS_KEY, SCORECARD_KEY } from "./query-keys";

/**
 * A picker passes `limit: 200`, the server's cap, to see past the default page of 50.
 * ponytail: a picker sees the newest 200; past that it needs search, not a bigger page.
 */
export function useProtocols(cursor?: string, limit?: number) {
  return useQuery({
    queryKey: [...PROTOCOLS_KEY, cursor ?? null, limit ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseProtocolResponse>({
        url: `${API_V1}/protocols`,
        method: "GET",
        params: { cursor, limit },
      }),
  });
}

export function useProtocol(id: string | undefined) {
  return useQuery({
    queryKey: [...PROTOCOL_KEY, id],
    queryFn: () =>
      customInstance<ProtocolResponse>({ url: `${API_V1}/protocols/${id}`, method: "GET" }),
    enabled: Boolean(id),
  });
}

/** The protocol's chemical-space map. Written once at training (or by the backfill). */
export function useProtocolChemicalSpace(id: string | undefined) {
  return useQuery({
    queryKey: [...PROTOCOL_KEY, id, "chemical-space"],
    queryFn: () =>
      customInstance<ChemicalSpaceResponse>({
        url: `${API_V1}/protocols/${id}/chemical-space`,
        method: "GET",
      }),
    enabled: Boolean(id),
    staleTime: mapStaleTime,
  });
}

/** One compound of the map, for a hover tooltip. Null index means nothing hovered. */
export function useProtocolMapCompound(id: string, index: number | null) {
  return useQuery({
    queryKey: [...PROTOCOL_KEY, id, "chemical-space", "compound", index],
    queryFn: async () =>
      (
        await customInstance<MapCompoundResponse[]>({
          url: `${API_V1}/protocols/${id}/chemical-space/compounds`,
          method: "GET",
          params: { indices: [index] },
        })
      )[0] ?? null,
    enabled: index !== null && index >= 0,
    staleTime: STALE_TIME.LONG,
  });
}

export function useScorecard(id: string | undefined) {
  return useQuery({
    queryKey: [...SCORECARD_KEY, id],
    queryFn: () =>
      customInstance<ScorecardResponse>({
        url: `${API_V1}/protocols/${id}/scorecard`,
        method: "GET",
      }),
    enabled: Boolean(id),
    // A published Protocol is immutable, and a draft's scorecard is written
    // once at training time. Neither ever changes under us.
    staleTime: STALE_TIME.LONG,
  });
}

/** Training. Returns a Run immediately; the Protocol appears when it finishes. */
export function useTrainProtocol() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: TrainProtocolBody) =>
      customInstance<RunResponse>({ url: `${API_V1}/protocols`, method: "POST", data }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PROTOCOLS_KEY });
    },
  });
}

/**
 * Poll a Run to completion.
 *
 * `refetchInterval` as a function of the query, deliberately: the alternative
 * -- a useEffect with a setTimeout chain and the job object in its dependency
 * array -- is how a sibling app once fired 7000+ requests at a single job id,
 * because the object was re-derived on every render. There is no effect here
 * and no dependency array, so that failure mode cannot occur.
 */
export function useRunPoll(runId: string | undefined) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => customInstance<RunResponse>({ url: `${API_V1}/runs/${runId}`, method: "GET" }),
    enabled: Boolean(runId),
    refetchInterval: pollInterval,
  });
}

/**
 * Publishing is irreversible; a second attempt is a 423 from the server.
 *
 * A 423 means someone published it first -- a stale view, not a failure -- so
 * this hook is silent to the global toast and reports every other failure itself.
 */
export function usePublishProtocol() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/protocols/${id}/publish`, method: "POST" }),
    meta: { silent: true },
    onSuccess: () => showSuccess("Protocol published. Anyone in this workspace can now run it."),
    onError: (error) => {
      if (error instanceof ApiError && (error.status === 423 || error.silent)) return;
      showError(error.message);
    },
    // Refetch whatever happened: after a 423 the cached draft is out of date too.
    onSettled: (_data, _error, id) => {
      queryClient.invalidateQueries({ queryKey: PROTOCOLS_KEY });
      queryClient.invalidateQueries({ queryKey: [...PROTOCOL_KEY, id] });
    },
  });
}

export function useDeleteProtocol() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/protocols/${id}`, method: "DELETE" }),
    // The dialog shows the error; no second toast.
    meta: { silent: true },
    onSuccess: () => {
      showSuccess("Protocol deleted.");
      // Lists, its training run, its sweep and its dataset's dialog all change.
      // Mark everything stale without refetching: the page being left would
      // otherwise refetch the protocol it just deleted and flash a 404.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}

/**
 * The Runs that belong to this Protocol -- its training Run, and every
 * prediction made with it.
 *
 * Migration 007 created `ix_runs_workspace_protocol_id` and documented it as
 * backing "the Protocol detail page's run history, and the only query this
 * column exists to serve". No query used it and this section did not exist;
 * both are now true.
 */
export function useProtocolRuns(protocolId: string | undefined) {
  return useQuery({
    queryKey: [...PROTOCOL_RUNS_KEY, protocolId],
    queryFn: () =>
      customInstance<PaginatedResponseRunResponse>({
        url: `${API_V1}/runs`,
        method: "GET",
        params: { protocol_id: protocolId },
      }),
    enabled: Boolean(protocolId),
    staleTime: STALE_TIME.SHORT,
  });
}
