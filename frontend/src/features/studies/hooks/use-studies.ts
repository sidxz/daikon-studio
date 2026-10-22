"use client";

import { API_V1, type ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import type { StartStudyRequest } from "@/shared/lib/api/model";
import { RUN_POLL_MS, STALE_TIME } from "@/shared/lib/query-defaults";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Study, StudyValue, Thread } from "../types";
import { STUDIES_KEY } from "./query-keys";

export function useStudies() {
  return useQuery({
    queryKey: STUDIES_KEY,
    queryFn: () => customInstance<Study[]>({ url: `${API_V1}/studies`, method: "GET" }),
  });
}

/** Polled while a step is in flight, so each one is seen to start and finish. */
export function useStudy(id: string | undefined) {
  return useQuery({
    queryKey: [...STUDIES_KEY, id],
    queryFn: () => customInstance<Study>({ url: `${API_V1}/studies/${id}`, method: "GET" }),
    enabled: Boolean(id),
    refetchInterval: (query) => (query.state.data?.running ? RUN_POLL_MS : false),
  });
}

export function useStartStudy() {
  const client = useQueryClient();
  return useMutation<Study, ApiError, { plan: StartStudyRequest; file?: File }>({
    mutationFn: ({ plan, file }) => {
      // Multipart so the compounds travel with the plan; no Content-Type header,
      // the browser sets it with the boundary.
      const form = new FormData();
      form.append("plan", JSON.stringify(plan));
      if (file) form.append("file", file);
      return customInstance<Study>({ url: `${API_V1}/studies`, method: "POST", data: form });
    },
    onSuccess: (study) => {
      client.setQueryData([...STUDIES_KEY, study.id], study);
      void client.invalidateQueries({ queryKey: STUDIES_KEY, exact: true });
    },
    meta: { silent: true },
  });
}

/**
 * Run one step, the one pressed. The server answers 202 with the study as it
 * stands, `running` set, and the poll in `useStudy` takes it from there.
 */
export function useRunStep(studyId: string) {
  const client = useQueryClient();
  return useMutation<Study, ApiError, number>({
    mutationFn: (position) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/steps/${position}/run`,
        method: "POST",
      }),
    onSuccess: (study) => {
      client.setQueryData([...STUDIES_KEY, studyId], study);
      void client.invalidateQueries({ queryKey: STUDIES_KEY, exact: true });
      void client.invalidateQueries({ queryKey: [...STUDIES_KEY, studyId], exact: true });
    },
  });
}

/** A value is written once and never changes, so it is read once. */
export function useStudyValue(studyId: string, valueId: string | undefined) {
  return useQuery({
    queryKey: [...STUDIES_KEY, studyId, "values", valueId],
    queryFn: () =>
      customInstance<StudyValue>({
        url: `${API_V1}/studies/${studyId}/values/${valueId}`,
        method: "GET",
      }),
    enabled: Boolean(valueId),
    staleTime: STALE_TIME.STATIC,
  });
}

/** Where a conversation lives: one stage, or the study itself when position is null. */
function chatUrl(studyId: string, position: number | null) {
  return position === null
    ? `${API_V1}/studies/${studyId}/chat`
    : `${API_V1}/studies/${studyId}/steps/${position}/chat`;
}

/** What has been said in one conversation. Null until anything has been. */
export function useStageThread(studyId: string, position: number | null) {
  return useQuery({
    queryKey: [...STUDIES_KEY, studyId, "chat", position],
    queryFn: () =>
      customInstance<Thread | null>({ url: chatUrl(studyId, position), method: "GET" }),
    staleTime: STALE_TIME.DEFAULT,
  });
}

/**
 * Say something to the agent at one stage, or open the conversation with null.
 *
 * The answer is the whole thread rather than the new turn, because the server
 * appends both sides and is the one that knows what survived the check.
 */
export function useSay(studyId: string, position: number | null) {
  const client = useQueryClient();
  return useMutation<Thread, ApiError, string | null>({
    mutationFn: (message) =>
      customInstance<Thread>({
        url: chatUrl(studyId, position),
        method: "POST",
        data: { message },
      }),
    onSuccess: (thread) => {
      client.setQueryData([...STUDIES_KEY, studyId, "chat", position], thread);
    },
    meta: { silent: true },
  });
}

/**
 * Make a change an agent proposed.
 *
 * Answers with the whole study because a proposal can add a step, and the step
 * list is what the scientist is about to look at. The conversation is refetched
 * too, so the proposal comes back marked as made.
 */
export function useApplyProposal(studyId: string) {
  const client = useQueryClient();
  return useMutation<Study, ApiError, string>({
    mutationFn: (proposalId) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/proposals/${proposalId}/apply`,
        method: "POST",
      }),
    onSuccess: (study) => {
      client.setQueryData([...STUDIES_KEY, studyId], study);
      void client.invalidateQueries({ queryKey: [...STUDIES_KEY, studyId, "chat"] });
    },
    meta: { silent: true },
  });
}
