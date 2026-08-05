"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type { CreatedRunnerResponse, RunnerResponse } from "@/shared/lib/api/model";
import { RUNNER_POLL_MS } from "@/shared/lib/query-defaults";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RUNNERS_KEY } from "./query-keys";

/** A plain list, no pagination wrapper -- the backend expects single-digit runners. */
export function useRunners() {
  return useQuery({
    queryKey: RUNNERS_KEY,
    queryFn: () => customInstance<RunnerResponse[]>({ url: `${API_V1}/runners`, method: "GET" }),
    refetchInterval: RUNNER_POLL_MS,
  });
}

/** Response carries the one-time token; the dialog is what makes sure it gets seen. */
export function useCreateRunner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; lanes: string[] }) =>
      customInstance<CreatedRunnerResponse>({ url: `${API_V1}/runners`, method: "POST", data }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: RUNNERS_KEY }),
  });
}

export function useRevokeRunner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/runners/${id}/revoke`, method: "POST" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: RUNNERS_KEY });
      showSuccess("Runner revoked");
    },
  });
}
