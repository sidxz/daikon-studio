"use client";

import { API_V1, ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import type {
  ChemicalSpaceResponse,
  MapCompoundResponse,
  PaginatedResponseProtocolResponse,
  PaginatedResponseRunResponse,
  ProtocolResponse,
  ProtocolStatus,
  RunResponse,
  ScorecardResponse,
  ScorecardToleranceResponse,
  TrainProtocolBody,
} from "@/shared/lib/api/model";
import { type Headline, headlines } from "@/shared/lib/headlines";
import {
  RUN_POLL_MS,
  RUN_RETRY_POLL_MS,
  STALE_TIME,
  isTerminal,
  mapStaleTime,
  pollInterval,
} from "@/shared/lib/query-defaults";
import { showError, showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { PROTOCOLS_KEY, PROTOCOL_KEY, PROTOCOL_RUNS_KEY, SCORECARD_KEY } from "./query-keys";

export interface ProtocolFilters {
  mine?: boolean;
  folderId?: string;
  q?: string;
  engineId?: string;
  status?: ProtocolStatus;
}

/**
 * A picker passes `limit: 200`, the server's cap, to see past the default page of 50.
 * ponytail: a picker sees the newest 200; past that it needs search, not a bigger page.
 */
export function useProtocols(cursor?: string, limit?: number, filters?: ProtocolFilters) {
  return useQuery({
    queryKey: [...PROTOCOLS_KEY, cursor ?? null, limit ?? null, filters ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseProtocolResponse>({
        url: `${API_V1}/protocols`,
        method: "GET",
        params: {
          cursor,
          limit,
          mine: filters?.mine || undefined,
          folder_id: filters?.folderId,
          q: filters?.q,
          engine_id: filters?.engineId,
          status: filters?.status,
        },
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

/** The complete catalogue for searchable run pickers and run-name resolution. */
export function useProtocolOptions() {
  return useQuery({
    queryKey: [...PROTOCOLS_KEY, "options"],
    staleTime: STALE_TIME.MEDIUM,
    queryFn: async ({ signal }) => {
      const protocols: ProtocolResponse[] = [];
      let cursor: string | undefined;
      const visited = new Set<string>();
      do {
        const page = await customInstance<PaginatedResponseProtocolResponse>({
          url: `${API_V1}/protocols`,
          method: "GET",
          params: { cursor, limit: 200 },
          signal,
        });
        protocols.push(...page.items);
        cursor = page.next_cursor ?? undefined;
        if (cursor && visited.has(cursor))
          throw new Error("Could not load all protocols. Try again.");
        if (cursor) visited.add(cursor);
      } while (cursor);
      return protocols;
    },
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
      customInstance<ScorecardResponse[]>({
        url: `${API_V1}/protocols/${id}/scorecard`,
        method: "GET",
      }),
    enabled: Boolean(id),
    // A published Protocol is immutable, and a draft's scorecard is written
    // once at training time. Neither ever changes under us.
    staleTime: STALE_TIME.LONG,
  });
}

export function useScorecardTolerance(id: string, target: string, tolerance: number | null) {
  return useQuery({
    queryKey: [...SCORECARD_KEY, id, "tolerance", target, tolerance],
    queryFn: ({ signal }) =>
      customInstance<ScorecardToleranceResponse>({
        url: `${API_V1}/protocols/${id}/scorecard/tolerance`,
        method: "GET",
        params: { target, tolerance },
        signal,
      }),
    enabled: tolerance !== null,
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

/**
 * Training runs still in flight. Between "Train" and the Protocol it produces
 * this run is the only record of the work, and the train form holds its id in
 * state alone, so closing that tab would otherwise lose the way back to it.
 *
 * Polls while any run is live, and refreshes the Protocols list when one
 * finishes so the new Protocol appears where the run just was. It calls the
 * list endpoint itself because `@/features/runs` imports this feature's barrel.
 * ponytail: only the newest page of training runs is checked; an active run
 * older than that page is missed. Add a status filter to `GET /runs` if that
 * ever happens.
 */
export function useActiveTrainingRuns() {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: [...PROTOCOL_RUNS_KEY, "active-training"],
    queryFn: () =>
      customInstance<PaginatedResponseRunResponse>({
        url: `${API_V1}/runs`,
        method: "GET",
        params: { kind: "training" },
      }),
    refetchInterval: (query) => {
      if (!query.state.data?.items.some((run) => !isTerminal(run.status))) return false;
      // A failed refetch with the list on screen is a blip, not the end of the watch.
      return query.state.status === "error" ? RUN_RETRY_POLL_MS : RUN_POLL_MS;
    },
  });
  const live = (data?.items ?? []).filter((run) => !isTerminal(run.status));

  const liveBefore = useRef(live.length);
  useEffect(() => {
    if (live.length < liveBefore.current) {
      queryClient.invalidateQueries({ queryKey: PROTOCOLS_KEY });
    }
    liveBefore.current = live.length;
  }, [live.length, queryClient]);

  return live;
}

/**
 * Each protocol's headline scores, read off its training run: one request for the
 * whole list, keyed under the protocols list so a finished training refreshes both.
 * ponytail: the newest 200 training runs only; an older protocol shows no score.
 * Upgrade: return the headline on the protocol itself.
 */
export function useTrainingHeadlines(): Map<string, Headline[]> {
  const { data } = useQuery({
    queryKey: [...PROTOCOLS_KEY, "training-headlines"],
    queryFn: () =>
      customInstance<PaginatedResponseRunResponse>({
        url: `${API_V1}/runs`,
        method: "GET",
        params: { kind: "training", limit: 200 },
      }),
  });
  return useMemo(() => {
    const byProtocol = new Map<string, Headline[]>();
    // Newest first, so the first run with scores is the protocol's latest.
    for (const run of data?.items ?? []) {
      const scores = headlines(run.metrics);
      if (run.protocol_id && scores.length > 0 && !byProtocol.has(run.protocol_id)) {
        byProtocol.set(run.protocol_id, scores);
      }
    }
    return byProtocol;
  }, [data]);
}
