"use client";

import { API_V1, type ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import type { ApproveRequest, RunRequest, SayRequest, SwapRequest } from "@/shared/lib/api/model";
import { RUN_POLL_MS, STALE_TIME } from "@/shared/lib/query-defaults";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Provider, Study, StudyValue, Thread } from "../types";
import { PROVIDERS_KEY, STUDIES_KEY } from "./query-keys";

/** Where a study's one conversation is cached. */
const threadKey = (studyId: string) => [...STUDIES_KEY, studyId, "chat"];

/**
 * Which model providers this deployment can reach. Fixed at deploy time, so the
 * long staleness tier. An empty list means no model is configured, which the page
 * says plainly rather than offering a button that always fails.
 */
export function useProviders() {
  return useQuery({
    queryKey: PROVIDERS_KEY,
    queryFn: () =>
      customInstance<Provider[]>({ url: `${API_V1}/studies/providers`, method: "GET" }),
    staleTime: STALE_TIME.LONG,
  });
}

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

/**
 * The one conversation of a study, from the agent's first questions on. A study
 * has a thread from its first message, so this is never null. Every write below
 * leaves it current; the page reads it again whenever the study's `updated_at`
 * moves, because a run writes turns while no request is open.
 */
export function useThread(studyId: string) {
  return useQuery({
    queryKey: threadKey(studyId),
    queryFn: () =>
      customInstance<Thread>({ url: `${API_V1}/studies/${studyId}/chat`, method: "GET" }),
  });
}

/**
 * What every write that answers with a study leaves behind. The answer is the
 * study as it now stands, so it goes straight into the cache. The thread may have
 * gained a turn the server wrote (an event, a reply) and the list shows when each
 * study last moved, neither of which the answer carries, so both are read again.
 */
function useAfterWrite() {
  const client = useQueryClient();
  return (study: Study) => {
    client.setQueryData([...STUDIES_KEY, study.id], study);
    void client.invalidateQueries({ queryKey: threadKey(study.id), exact: true });
    void client.invalidateQueries({ queryKey: STUDIES_KEY, exact: true });
  };
}

/**
 * The first message of a study. It opens a draft and the agent's first turn --
 * usually questions -- and answers with the study only; the thread is read with
 * `useThread`. `silent` because a refusal here is worth a sentence on the page --
 * which variable to set, which file could not be read -- and a toast would be a
 * second copy.
 */
export function useOpenStudy() {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, { message: string; provider?: string; files?: File[] }>({
    mutationFn: ({ message, provider, files }) => {
      // Multipart, so the words and the files arrive together. No Content-Type
      // header: the browser sets it with the boundary, and naming it ourselves
      // produces a body the server cannot parse.
      const form = new FormData();
      form.append("message", message);
      if (provider) form.append("provider", provider);
      for (const file of files ?? []) form.append("files", file);
      return customInstance<Study>({ url: `${API_V1}/studies`, method: "POST", data: form });
    },
    onSuccess: written,
    meta: { silent: true },
  });
}

/** Add files to a study that is already open. What was read from them is in `brought`. */
export function useAttachFiles(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, File[]>({
    mutationFn: (files) => {
      const form = new FormData();
      for (const file of files) form.append("files", file);
      return customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/files`,
        method: "POST",
        data: form,
      });
    },
    onSuccess: written,
  });
}

/**
 * Start a drafted study: the scientist has read the plan and says go. Nothing
 * runs; it only lets the steps be run. `silent` because the page shows a refusal
 * beside the button.
 */
export function useStartStudy(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, void>({
    mutationFn: () =>
      customInstance<Study>({ url: `${API_V1}/studies/${studyId}/start`, method: "POST" }),
    onSuccess: written,
    meta: { silent: true },
  });
}

/**
 * Run one step, the one pressed. The server answers 202 with the study as it
 * stands, `running` set, and the poll in `useStudy` takes it from there.
 */
export function useRunStep(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, string>({
    mutationFn: (stepId) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/steps/${stepId}/run`,
        method: "POST",
      }),
    onSuccess: written,
  });
}

/**
 * Run on, driven by the server: from `start` (default: the next step waiting) to
 * `until` (default: the next step that stops for a look). The route requires a
 * body, so "run to the next stop" is `{}`. 202, like `useRunStep`.
 */
export function useRunTo(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, RunRequest>({
    mutationFn: (range) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/run`,
        method: "POST",
        data: range,
      }),
    onSuccess: written,
  });
}

/**
 * Put one of a step's alternatives in its place. The step keeps its id; a setting
 * the new tool needs and the old one lacked comes back as the step's `problem`.
 */
export function useSwapTool(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, { stepId: string; toolId: string }>({
    mutationFn: ({ stepId, toolId }) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/steps/${stepId}/tool`,
        method: "POST",
        data: { tool_id: toolId } satisfies SwapRequest,
      }),
    onSuccess: written,
  });
}

/**
 * Choose one of a step's candidates, by the rank the step reported it under. The
 * server records it as a value the scientist asserted, beside the candidates.
 */
export function useApprove(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, { stepId: string; candidate: number }>({
    mutationFn: ({ stepId, candidate }) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/steps/${stepId}/approve`,
        method: "POST",
        data: { candidate } satisfies ApproveRequest,
      }),
    onSuccess: written,
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

/**
 * Say something to the agent: a message, the answers to its questions, or both,
 * and optionally which step it is about. Neither a message nor answers asks it
 * for the turn it owes.
 *
 * The answer is the whole thread rather than the new turn, because the server
 * appends both sides and is the one that knows what survived the check. The study
 * is not in the answer, and a turn can change it -- an edit to the plan, a run
 * begun -- so it is read again, and with it the list.
 */
export function useSay(studyId: string) {
  const client = useQueryClient();
  return useMutation<Thread, ApiError, SayRequest>({
    mutationFn: (body) =>
      customInstance<Thread>({
        url: `${API_V1}/studies/${studyId}/chat`,
        method: "POST",
        data: body,
      }),
    onSuccess: (thread) => {
      client.setQueryData(threadKey(studyId), thread);
      void client.invalidateQueries({ queryKey: [...STUDIES_KEY, studyId], exact: true });
      void client.invalidateQueries({ queryKey: STUDIES_KEY, exact: true });
    },
    meta: { silent: true },
  });
}

/**
 * Make a change an agent proposed.
 *
 * Answers with the whole study because a proposal can add a step or start a run,
 * and the step list is what the scientist is about to look at. The thread is read
 * again too, so the proposal comes back marked as made.
 */
export function useApplyProposal(studyId: string) {
  const written = useAfterWrite();
  return useMutation<Study, ApiError, string>({
    mutationFn: (proposalId) =>
      customInstance<Study>({
        url: `${API_V1}/studies/${studyId}/proposals/${proposalId}/apply`,
        method: "POST",
      }),
    onSuccess: written,
    meta: { silent: true },
  });
}
