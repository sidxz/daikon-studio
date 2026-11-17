import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PlanTerms } from "./plan-terms";

const turns = vi.hoisted(() => ({ value: [] as unknown[] }));
const steps = vi.hoisted(() => ({ value: [] as unknown[] }));

vi.mock("../hooks/use-studies", () => ({
  useThread: () => ({ data: { turns: turns.value } }),
  useStudy: () => ({ data: { steps: steps.value } }),
}));

const CHECK = "A candidate contains the annotated catalytic residues.";

function point(state: string, label: string) {
  return { step: null, at: "check 1", label, evidence: "check", cites: [CHECK], state };
}

describe("where a criterion stands", () => {
  it("marks the criterion from the standing the agent declared, with its reason and step", () => {
    steps.value = [{ id: "s1", position: 3, name: "Find binding pockets" }];
    turns.value = [
      { about: "s1", points: [point("met", "an early reading")] },
      { about: "s1", points: [point("not_met", "no candidate lists Tyr158 or Lys165")] },
    ];

    render(<PlanTerms studyId="x" checks={[CHECK]} stopConditions={[]} />);

    // The latest word wins: a criterion can stop holding once a later step has run.
    expect(screen.getByText(/no candidate lists Tyr158/)).toBeInTheDocument();
    expect(screen.queryByText("an early reading")).not.toBeInTheDocument();
    expect(screen.getByText(/step 4/)).toBeInTheDocument();
    expect(screen.getByText("✕")).toBeInTheDocument();
  });

  it("leaves a criterion nothing has spoken to unmarked", () => {
    steps.value = [];
    turns.value = [];

    render(<PlanTerms studyId="x" checks={[CHECK]} stopConditions={[]} />);

    expect(screen.getByText(CHECK)).toBeInTheDocument();
    expect(screen.getByText("·")).toBeInTheDocument();
    expect(screen.queryByText("✓")).not.toBeInTheDocument();
  });
});
