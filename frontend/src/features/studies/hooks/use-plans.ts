"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { Plan, Provider, Questions } from "../types";
import { PROVIDERS_KEY } from "./query-keys";

/**
 * Which model providers this deployment can reach. Fixed at deploy time, so the
 * long staleness tier. An empty list means no planner is configured, which the
 * page says plainly rather than offering a button that always fails.
 */
export function useProviders() {
  return useQuery({
    queryKey: PROVIDERS_KEY,
    queryFn: () => customInstance<Provider[]>({ url: `${API_V1}/plans/providers`, method: "GET" }),
    staleTime: STALE_TIME.LONG,
  });
}

/**
 * Drafting writes nothing, so there is nothing to invalidate. `silent` because a
 * refusal here is worth a paragraph on the page -- which variable to set, or what
 * the model did wrong -- and a toast would be both too small and a second copy.
 */
/**
 * The first half: what the planner needs to know, before it has planned
 * anything. Separate from drafting so the plan is an answer to something rather
 * than a guess that already looks settled.
 */
/**
 * Round one: the questions the plan genuinely cannot be written without. A model
 * that can defend a default is told to use it, so this often comes back empty,
 * and empty is the good case -- it means the data could be read well enough to
 * proceed and nobody has to fill in a form.
 */
export function useAskFirst() {
  return useMutation({
    mutationFn: ({
      goal,
      provider,
      file,
      answers,
    }: {
      goal: string;
      provider?: string;
      file?: File;
      answers?: { ask: string; answer: string }[];
    }) => {
      const form = new FormData();
      form.append("goal", goal);
      if (provider) form.append("provider", provider);
      if (file) form.append("file", file);
      return customInstance<Questions>({
        url: `${API_V1}/plans/questions`,
        method: "POST",
        data: form,
      });
    },
    meta: { silent: true },
  });
}

export function useDraftPlan() {
  return useMutation({
    mutationFn: ({
      goal,
      provider,
      file,
      answers,
    }: {
      goal: string;
      provider?: string;
      file?: File;
      answers?: { ask: string; answer: string }[];
    }) => {
      // Multipart, so the goal and the data arrive together. No Content-Type
      // header: the browser sets it with the boundary, and naming it ourselves
      // produces a body the server cannot parse.
      const form = new FormData();
      form.append("goal", goal);
      if (provider) form.append("provider", provider);
      if (file) form.append("file", file);
      // A form field rather than a JSON body: the rest of the request is
      // multipart so the file can travel with it, and the plan is written
      // against both the shape of the data and what was said about it.
      if (answers?.length) form.append("answers", JSON.stringify(answers));
      return customInstance<Plan>({ url: `${API_V1}/plans`, method: "POST", data: form });
    },
    meta: { silent: true },
  });
}
