import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Scorecards } from "./protocol-detail";

// The scorecard body draws charts and the average has its own tests; which tab
// shows what is the only thing under test here.
vi.mock("./scorecard-view", () => ({
  ScorecardView: ({ scorecard }: { scorecard: ScorecardResponse }) => (
    <div data-testid="scorecard">{scorecard.target}</div>
  ),
}));
vi.mock("./scorecard-compounds", () => ({
  ScorecardCompounds: ({ scorecard }: { scorecard: ScorecardResponse }) => (
    <div data-testid="compounds">{scorecard.target}</div>
  ),
}));
vi.mock("./scorecard-aggregate", () => ({
  ScorecardAggregate: ({ scorecards }: { scorecards: ScorecardResponse[] }) => (
    <div data-testid="aggregate">{scorecards.length} targets</div>
  ),
}));

const card = (target: string, joint_model: boolean) =>
  ({ target, joint_model }) as unknown as ScorecardResponse;

const TWO = [card("solubility", false), card("reactive", false)];

describe("Scorecards", () => {
  it("opens on the average across targets, with a tab per target after it", () => {
    render(<Scorecards scorecards={TWO} />);

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "All targets",
      "solubility",
      "reactive",
    ]);
    expect(screen.getByTestId("aggregate")).toHaveTextContent("2 targets");
    expect(screen.queryByTestId("scorecard")).not.toBeInTheDocument();
  });

  it("says the models were trained one per target", () => {
    render(<Scorecards scorecards={TWO} />);

    expect(
      screen.getByText(
        "2 separate models, one per target, trained on the same compounds and split.",
      ),
    ).toBeInTheDocument();
  });

  it("says one model learned every target when the training was joint", () => {
    render(<Scorecards scorecards={[card("a", true), card("b", true)]} />);

    expect(screen.getByText("One model learned all 2 targets jointly.")).toBeInTheDocument();
  });

  it("renders a single target as a bare scorecard, without tabs or an average", () => {
    render(<Scorecards scorecards={[card("solubility", false)]} />);

    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("aggregate")).not.toBeInTheDocument();
    expect(screen.getByTestId("scorecard")).toHaveTextContent("solubility");
  });

  it("offers no average over test compounds, where there is nothing to average", () => {
    render(<Scorecards scorecards={TWO} view="compounds" />);

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "solubility",
      "reactive",
    ]);
    expect(screen.getByTestId("compounds")).toHaveTextContent("solubility");
  });

  it("shows the chosen target", () => {
    render(<Scorecards scorecards={TWO} selectedTarget="reactive" />);

    expect(screen.getByTestId("scorecard")).toHaveTextContent("reactive");
  });

  it("falls back to its first tab when the chosen target is not one of these", () => {
    // The two views share one selection, and only one of them has an average
    // tab: choosing it and switching to the test compounds must not blank out.
    render(<Scorecards scorecards={TWO} selectedTarget="__average__" view="compounds" />);

    expect(screen.getByTestId("compounds")).toHaveTextContent("solubility");
  });

  it("keeps the average reachable when a target is named like its tab", () => {
    render(<Scorecards scorecards={[card("__average__", false), card("reactive", false)]} />);

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "All targets",
      "__average__",
      "reactive",
    ]);
    expect(screen.getByTestId("aggregate")).toBeInTheDocument();
    expect(screen.queryByTestId("scorecard")).not.toBeInTheDocument();
  });
});
