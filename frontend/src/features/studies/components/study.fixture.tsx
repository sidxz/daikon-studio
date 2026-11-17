/**
 * What the study tests share: the server's shapes with plain defaults, and a table that answers
 * each request. Every test file mocks `customInstance` itself (a mock is hoisted per file); the
 * helpers here read that mock.
 */

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { vi } from "vitest";
import type { Study, StudyStep, Turn } from "../types";

export function step(overrides: Partial<StudyStep> = {}): StudyStep {
  return {
    id: "st1",
    position: 0,
    tool_id: "standardize",
    name: "Standardize structures",
    purpose: "Clean up.",
    params: {},
    gated: false,
    always: false,
    expanded_from: null,
    status: "pending",
    error: null,
    problem: null,
    started_at: null,
    finished_at: null,
    outputs: [],
    judgment: false,
    alternatives: [],
    ...overrides,
  };
}

/** A started study unless told otherwise. */
export function study(overrides: Partial<Study> = {}): Study {
  return {
    id: "s1",
    goal: "Find the binding pocket of carbonic anhydrase II, PDB 1CA2",
    restated_goal: "Locate the cavity where ligands bind in carbonic anhydrase II.",
    checks: [],
    stop_conditions: [],
    status: "open",
    running: false,
    started_at: "2026-10-22T00:00:00Z",
    mode: "run",
    amended_after_steps: null,
    updated_at: "2026-10-22T00:00:00Z",
    created_at: "2026-10-22T00:00:00Z",
    steps: [step()],
    brought: [],
    answers: [],
    ...overrides,
  };
}

export function turn(overrides: Partial<Turn> = {}): Turn {
  return {
    role: "agent",
    kind: "reply",
    say: "",
    about: null,
    questions: [],
    points: [],
    proposals: [],
    unresolved: [],
    actions: [],
    looked: [],
    dropped: 0,
    silent: false,
    written_by: null,
    at: "2026-10-22T00:00:00Z",
    ...overrides,
  };
}

/**
 * Answer requests from a table keyed "METHOD /path", the path after the API prefix. A function
 * is called for its answer, so a test can change what the server says after a write.
 */
export function route(table: Record<string, unknown>) {
  vi.mocked(customInstance).mockImplementation(async ({ url, method }) => {
    const key = `${method} ${url.replace(API_V1, "")}`;
    if (!(key in table)) throw new Error(`No answer for ${key}`);
    const answer = table[key];
    return typeof answer === "function" ? answer() : answer;
  });
}

/** The requests made so far to one method and path. */
export function requests(method: string, path: string) {
  return vi
    .mocked(customInstance)
    .mock.calls.map(([config]) => config)
    .filter((config) => config.method === method && config.url === `${API_V1}${path}`);
}

export function Wrapper({ children }: { children: ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
