"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type {
  PaginatedResponseProtocolResponse,
  ProtocolResponse,
  RunResponse,
  ScorecardResponse,
  TrainProtocolBody,
} from "@/shared/lib/api/model";
import { RUN_POLL_MS, STALE_TIME } from "@/shared/lib/query-defaults";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PROTOCOLS_KEY, PROTOCOL_KEY, SCORECARD_KEY } from "./query-keys";

const TERMINAL = new Set(["ready", "failed", "cancelled"]);

export function isTerminal(status: string | undefined): boolean {
  return status !== undefined && TERMINAL.has(status);
}

export function useProtocols(cursor?: string) {
  return useQuery({
    queryKey: [...PROTOCOLS_KEY, cursor ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseProtocolResponse>({
        url: `${API_V1}/protocols`,
        method: "GET",
        params: cursor ? { cursor } : undefined,
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
    refetchInterval: (query) => (isTerminal(query.state.data?.status) ? false : RUN_POLL_MS),
  });
}

/** Publishing is irreversible; a second attempt is a 423 from the server. */
export function usePublishProtocol() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/protocols/${id}/publish`, method: "POST" }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: PROTOCOLS_KEY });
      queryClient.invalidateQueries({ queryKey: [...PROTOCOL_KEY, id] });
      showSuccess("Protocol published — anyone in this workspace can run it now");
    },
  });
}
