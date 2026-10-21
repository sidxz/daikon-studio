/**
 * What the plan page must never get wrong.
 *
 * A plan is a proposal, and this page exists to be argued with. The things that
 * would make it dishonest: hiding that the plan cannot run, hiding what the
 * planner did not understand, or presenting a composite as one opaque step when
 * the record keeps several.
 */

import { PlanComposer } from "@/features/planner";
import type { Plan, Questions } from "@/features/planner";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

function plan(overrides: Partial<Plan> = {}): Plan {
  return {
    goal: "g",
    restated_goal: "Understand what drives activity in this series.",
    questions: [],
    steps: [
      {
        tool_id: "standardize",
        name: "Standardize structures",
        purpose: "Clean up.",
        gated: true,
        expanded_from: null,
        always: false,
      },
    ],
    checks: [],
    stop_conditions: [],
    skipped: [],
    problems: [],
    written_by: "anthropic/claude-opus-5-5",
    questions_left: 7,
    ...overrides,
  };
}

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function questions(overrides: Partial<Questions> = {}): Questions {
  return {
    goal: "g",
    restated_goal: "Understand what drives activity in this series.",
    questions: [
      {
        ask: "Which column is the readout that matters?",
        why: "Every effect size is computed against one readout.",
        options: ["IC50_uM", "Ki_nM"],
      },
    ],
    written_by: "anthropic/claude-opus-5-5",
    questions_left: 9,
    ...overrides,
  };
}

function serve(providers: unknown, drafted?: Plan, asked?: Questions) {
  vi.mocked(customInstance).mockImplementation((config: { url: string }) => {
    if (config.url.endsWith("/plans/providers")) return Promise.resolve(providers);
    if (config.url.endsWith("/plans/questions")) return Promise.resolve(asked ?? questions());
    return Promise.resolve(drafted ?? plan());
  });
}

async function draft(goal: string) {
  fireEvent.change(await screen.findByLabelText(/What do you want to find out/), {
    target: { value: goal },
  });
  fireEvent.click(screen.getByRole("button", { name: "Draft a plan" }));
  // The first press only asks. Planning is the second.
  fireEvent.click(await screen.findByRole("button", { name: "Draft the plan" }));
}

describe("the plan page", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("says which variable to set when no provider is configured", async () => {
    serve([]);
    render(<PlanComposer />, { wrapper: Wrapper });

    expect(
      await screen.findByText(/No model provider is configured on this server/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Draft a plan" })).toBeDisabled();
  });

  it("will not draft on a goal too thin to plan from", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }]);
    render(<PlanComposer />, { wrapper: Wrapper });

    fireEvent.change(await screen.findByLabelText(/What do you want to find out/), {
      target: { value: "SAR" },
    });

    expect(screen.getByRole("button", { name: "Draft a plan" })).toBeDisabled();
    expect(screen.getByText(/Give a little more detail/)).toBeInTheDocument();
  });

  it("leads with what the planner could not settle on its own", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({
        questions: [{ ask: "Which column is the activity?", why: "Three columns could be." }],
      }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText("Which column is the activity?")).toBeInTheDocument();
    expect(screen.getByText("Three columns could be.")).toBeInTheDocument();
  });

  it("says plainly when the plan could not run as written", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({ problems: [{ step: 2, message: "There is no tool called 'magic_sar'." }] }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText(/This plan could not run as written/)).toBeInTheDocument();
    expect(screen.getByText(/no tool called 'magic_sar'/)).toBeInTheDocument();
  });

  it("groups a composite's steps without hiding them", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({
        steps: [
          {
            tool_id: "decompose_rgroups",
            name: "Decompose R groups",
            purpose: "Lay out the SAR.",
            gated: false,
            expanded_from: "build_sar_table",
            always: false,
          },
          {
            tool_id: "name_substituents",
            name: "Name substituents",
            purpose: "Lay out the SAR.",
            gated: true,
            expanded_from: "build_sar_table",
            always: false,
          },
        ],
      }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText(/One step: build sar table/)).toBeInTheDocument();
    // Both parts still shown: the chemist chose one thing, the record keeps two.
    expect(screen.getByText("Decompose R groups")).toBeInTheDocument();
    expect(screen.getByText("Name substituents")).toBeInTheDocument();
    expect(screen.getAllByText("stops for you")).toHaveLength(1);
  });

  it("names the model that wrote the plan", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }]);
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText(/anthropic\/claude-opus-5-5/)).toBeInTheDocument();
  });
});

describe("asking before planning", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("asks first and plans nothing", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }]);
    render(<PlanComposer />, { wrapper: Wrapper });
    fireEvent.change(await screen.findByLabelText(/What do you want to find out/), {
      target: { value: "Work out what is driving activity in this series." },
    });

    fireEvent.click(screen.getByRole("button", { name: "Draft a plan" }));

    expect(await screen.findByText(/Help Daikon Studio Understand More/)).toBeInTheDocument();
    expect(screen.getByText(/Nothing has been planned yet/)).toBeInTheDocument();
    // No plan on screen until the second press.
    expect(screen.queryByText("The plan")).toBeNull();
  });

  it("offers the model's options plus one it never produces", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }]);
    render(<PlanComposer />, { wrapper: Wrapper });
    fireEvent.change(await screen.findByLabelText(/What do you want to find out/), {
      target: { value: "Work out what is driving activity in this series." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Draft a plan" }));
    await screen.findByText(/Help Daikon Studio Understand More/);

    expect(screen.getByRole("button", { name: "IC50_uM" })).toBeInTheDocument();
    // Always available, never from the model: it is the answer most likely to
    // be true and least likely to be offered.
    expect(screen.getByRole("button", { name: "Not sure" })).toBeInTheDocument();
  });

  it("sends what was answered, and says what was not", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }]);
    render(<PlanComposer />, { wrapper: Wrapper });
    fireEvent.change(await screen.findByLabelText(/What do you want to find out/), {
      target: { value: "Work out what is driving activity in this series." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Draft a plan" }));
    await screen.findByText(/Help Daikon Studio Understand More/);

    expect(screen.getByText(/1 of 1 unanswered/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Ki_nM" }));
    expect(await screen.findByText("All answered.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Draft the plan" }));

    await waitFor(() => {
      const posts = vi
        .mocked(customInstance)
        .mock.calls.filter(([config]) => (config as { url: string }).url.endsWith("/plans"));
      expect(posts).toHaveLength(1);
      const form = (posts[0][0] as { data: FormData }).data;
      expect(JSON.parse(form.get("answers") as string)).toEqual([
        { ask: "Which column is the readout that matters?", answer: "Ki_nM" },
      ]);
    });
  });
});

describe("refining a finished plan", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  const refinable = () =>
    plan({
      questions_left: 4,
      questions: [
        {
          ask: "Call out the two compounds with no cell activity?",
          why: "Costs nothing and changes nothing in the analysis.",
          options: ["Yes, call them out", "No, drop MIC entirely"],
          default: "Yes, call them out",
        },
      ],
    });

  it("offers the planner's own answer already filled in", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }], refinable());
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText("Change any of these")).toBeInTheDocument();
    const field = screen.getByLabelText(/Call out the two compounds/);
    expect(field).toHaveValue("Yes, call them out");
    // Nothing differs yet, so redrawing the plan would spend a minute to be
    // told the same thing.
    expect(screen.getByRole("button", { name: "Update the plan" })).toBeDisabled();
  });

  it("lets you disagree and redraw the plan", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }], refinable());
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");
    await screen.findByText("Change any of these");

    fireEvent.click(screen.getByRole("button", { name: "No, drop MIC entirely" }));
    const update = screen.getByRole("button", { name: "Update the plan" });
    await waitFor(() => expect(update).toBeEnabled());
    fireEvent.click(update);

    await waitFor(() => {
      const posts = vi
        .mocked(customInstance)
        .mock.calls.filter(([c]) => (c as { url: string }).url.endsWith("/plans"));
      const last = (posts.at(-1)?.[0] as { data: FormData }).data;
      const sent = JSON.parse(last.get("answers") as string);
      expect(sent).toContainEqual({
        ask: "Call out the two compounds with no cell activity?",
        answer: "No, drop MIC entirely",
      });
    });
  });

  it("says when the questions are used up", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({ ...refinable(), questions_left: 0 }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText(/No more questions will be raised/)).toBeInTheDocument();
  });

  it("does not drop a plan question the planner could not default", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({
        questions: [
          {
            ask: "Which batch is the reference?",
            why: "Nothing settles it.",
            options: [],
            default: null,
          },
        ],
      }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");

    expect(await screen.findByText("Worth settling")).toBeInTheDocument();
    expect(screen.getByLabelText(/Which batch is the reference/)).toHaveValue("");
  });
});

describe("showing what was assumed", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  const blended = () =>
    plan({
      questions: [
        {
          ask: "What counts as wrecking solubility?",
          why: "The verdict is a function of the cutoffs.",
          options: ["Solubility >=50 uM", "Solubility >=100 uM"],
          // The planner's answer is frequently a blend of two options,
          // qualified, so it matches none of them exactly.
          default: "Solubility >=50 uM as a hard flag, plus a fold-change flag.",
        },
      ],
    });

  it("puts an assumption that matches no option into the row, highlighted", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }], blended());
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");
    await screen.findByText("Change any of these");

    const assumption = screen.getByRole("button", {
      name: /Solubility >=50 uM as a hard flag, plus a fold-change flag\./,
    });
    expect(assumption).toHaveAttribute("aria-pressed", "true");
    // The options it was blended from are still offered, unselected.
    expect(screen.getByRole("button", { name: "Solubility >=100 uM" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(
      screen.getByText(/what Daikon Studio assumed, and what the plan below already does/),
    ).toBeInTheDocument();
  });

  it("says so once you have changed it, and lets you go back", async () => {
    serve([{ provider: "anthropic", model: "claude-opus-5-5" }], blended());
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");
    await screen.findByText("Change any of these");

    fireEvent.click(screen.getByRole("button", { name: "Solubility >=100 uM" }));

    expect(await screen.findByText(/Changed from what Daikon Studio assumed/)).toBeInTheDocument();
    // The assumption stays in the row, one click away.
    const assumption = screen.getByRole("button", {
      name: /Solubility >=50 uM as a hard flag, plus a fold-change flag\./,
    });
    expect(assumption).toHaveAttribute("aria-pressed", "false");
  });

  it("does not duplicate an assumption that is already one of the options", async () => {
    serve(
      [{ provider: "anthropic", model: "claude-opus-5-5" }],
      plan({
        questions: [
          {
            ask: "Run the pair analysis once or three times?",
            why: "Three passes give intervals on every endpoint.",
            options: ["Once on pIC50", "Three passes"],
            default: "Once on pIC50",
          },
        ],
      }),
    );
    render(<PlanComposer />, { wrapper: Wrapper });
    await draft("Work out what is driving activity in this series.");
    await screen.findByText("Change any of these");

    expect(screen.getAllByRole("button", { name: "Once on pIC50" })).toHaveLength(1);
  });
});
