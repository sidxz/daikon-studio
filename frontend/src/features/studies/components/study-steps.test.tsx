/**
 * What a study page must never get wrong: which step a press runs, and what it
 * says a step produced. One play control per step and nothing that runs several;
 * a result shown only once it has been read.
 */

import { StudySteps } from "@/features/studies";
import type { Study, StudyStep, StudyValue } from "@/features/studies";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));
vi.mock("@/shared/components/chemistry/structure-thumbnail", () => ({
  StructureThumbnail: () => null,
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

function step(overrides: Partial<StudyStep> = {}): StudyStep {
  return {
    position: 0,
    tool_id: "standardize",
    name: "Standardize structures",
    purpose: "Clean up.",
    params: {},
    gated: false,
    always: false,
    discussable: false,
    expanded_from: null,
    status: "pending",
    error: null,
    started_at: null,
    finished_at: null,
    outputs: [],
    ...overrides,
  };
}

function study(overrides: Partial<Study> = {}): Study {
  return {
    id: "s1",
    goal: "Work out what drives activity.",
    restated_goal: "Understand what drives activity in this series.",
    checks: [],
    stop_conditions: [],
    status: "open",
    running: false,
    steps: [step(), step({ position: 1, tool_id: "property_profile", name: "Profile properties" })],
    brought: [],
    answers: [],
    planning: {},
    created_at: "2026-10-21T00:00:00Z",
    ...overrides,
  };
}

const table: StudyValue = {
  id: "v1",
  key: "compounds",
  kind: "compound_table",
  summary: { rows: 4, columns: ["id", "SMILES"] },
  data: [
    { id: "c1", SMILES: "CCO" },
    { id: "c2", SMILES: "CCN" },
    { id: "c3", SMILES: "CCC" },
    { id: "c4", SMILES: "CCCl" },
  ],
};

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function serve(served: Study, value: StudyValue = table) {
  vi.mocked(customInstance).mockImplementation((config: { url: string }) => {
    // The panel asks for its conversation on every render of a study.
    if (config.url.includes("/chat")) return Promise.resolve(null);
    if (config.url.includes("/values/")) return Promise.resolve(value);
    return Promise.resolve(served);
  });
}

function runs() {
  return vi
    .mocked(customInstance)
    .mock.calls.filter(([config]) => (config as { url: string }).url.endsWith("/run"));
}

function valueFetches() {
  return vi
    .mocked(customInstance)
    .mock.calls.filter(([config]) => (config as { url: string }).url.includes("/values/"));
}

describe("the study page", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("keeps the conversation beside the results, not inside the step", async () => {
    // A claim is only checkable while the table it points at is still on screen,
    // so the panel is docked once rather than repeated under each step.
    const discussable = study({
      steps: [
        step({ status: "done" }),
        step({
          position: 1,
          tool_id: "find_pockets",
          name: "Find binding pockets",
          status: "done",
          discussable: true,
        }),
      ],
    });
    vi.mocked(customInstance).mockImplementation((config: { url: string }) => {
      if (config.url.includes("/chat")) return Promise.resolve(null);
      if (config.url.includes("/values/")) return Promise.resolve(table);
      return Promise.resolve(discussable);
    });
    render(<StudySteps studyId="s1" />, { wrapper: Wrapper });

    // The step offers the conversation; it does not hold it. It reads "Talking"
    // here because this stage is the one the panel already has open.
    expect(
      await screen.findByRole("button", { name: /talk(ing)? about this result/i }),
    ).toBeVisible();
    // Docked once, however many steps could be discussed.
    const boxes = await screen.findAllByPlaceholderText(/ask about this result/i);
    expect(boxes).toHaveLength(1);

    // Pressing it must land somewhere even when that stage is already the one
    // on show, or the button reads as broken.
    fireEvent.click(screen.getByRole("button", { name: /talk about this result/i }));
    expect(await screen.findByPlaceholderText(/ask about this result/i)).toHaveFocus();
  });

  it("offers one play control per step, and runs only the one pressed", async () => {
    serve(study());
    render(<StudySteps studyId="s1" />, { wrapper: Wrapper });

    expect(await screen.findByText("Standardize structures")).toBeInTheDocument();
    expect(screen.getByText("Profile properties")).toBeInTheDocument();
    expect(screen.getAllByText("Not run yet")).toHaveLength(2);
    const plays = screen.getAllByRole("button", { name: "Run this step" });
    expect(plays).toHaveLength(2);
    expect(screen.getAllByRole("button")).toHaveLength(2);
    expect(screen.queryByText(/Run to the next stop|Continue/)).not.toBeInTheDocument();

    fireEvent.click(plays[1]);

    await vi.waitFor(() => expect(runs()).toHaveLength(1));
    expect((runs()[0][0] as { url: string; method: string }).url).toMatch(
      /\/studies\/s1\/steps\/1\/run$/,
    );
    expect((runs()[0][0] as { method: string }).method).toBe("POST");
  });

  it("shows what a gated step produced, and offers to run it again", async () => {
    serve(
      study({
        steps: [
          step({
            status: "done",
            gated: true,
            outputs: [
              {
                id: "v1",
                key: "compounds",
                kind: "compound_table",
                summary: { rows: 4, columns: ["id", "SMILES"] },
                produced_by: 0,
                asserted_by: null,
              },
            ],
          }),
          step({ position: 1, tool_id: "property_profile", name: "Profile properties" }),
        ],
      }),
    );
    render(<StudySteps studyId="s1" />, { wrapper: Wrapper });

    expect(await screen.findByRole("button", { name: "Run this step again" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Run this step" })).toBeEnabled();
    expect(await screen.findByRole("columnheader", { name: "id" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "SMILES" })).toBeInTheDocument();
    expect(screen.getByText("c1")).toBeInTheDocument();
    // Header row plus the four compounds.
    expect(screen.getAllByRole("row")).toHaveLength(5);
    expect(valueFetches()).toHaveLength(1);
    expect((valueFetches()[0][0] as { url: string }).url).toMatch(/\/studies\/s1\/values\/v1$/);
  });

  it("will not start a second run while one is going", async () => {
    serve(study({ running: true, steps: [step({ status: "running" }), step({ position: 1 })] }));
    render(<StudySteps studyId="s1" />, { wrapper: Wrapper });

    const buttons = await screen.findAllByRole("button");
    expect(buttons).toHaveLength(2);
    for (const button of buttons) expect(button).toBeDisabled();
    // The status line and the control both say so.
    expect(screen.getAllByText("Running")).toHaveLength(2);
  });

  it("shows why a step failed, and offers to run it again", async () => {
    serve(
      study({
        status: "failed",
        steps: [
          step({ status: "done" }),
          step({
            position: 1,
            name: "Profile properties",
            status: "failed",
            error: "Needs compound table, which nothing produced.",
          }),
        ],
      }),
    );
    render(<StudySteps studyId="s1" />, { wrapper: Wrapper });

    expect(
      await screen.findByText("Needs compound table, which nothing produced."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run this step" })).toBeEnabled();
  });
});
