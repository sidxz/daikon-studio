import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Scorecards } from "./protocol-detail";

// The scorecard body draws charts; which target each tab shows is what matters here.
vi.mock("./scorecard-view", () => ({
  ScorecardView: ({ scorecard }: { scorecard: ScorecardResponse }) => (
    <div data-testid="scorecard">{scorecard.target}</div>
  ),
}));

const card = (target: string, joint_model: boolean) =>
  ({ target, joint_model }) as unknown as ScorecardResponse;

describe("Scorecards", () => {
  it("opens one tab per target and says the models were trained one per target", () => {
    render(<Scorecards scorecards={[card("solubility", false), card("reactive", false)]} />);

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "solubility",
      "reactive",
    ]);
    expect(
      screen.getByText(
        "2 separate models, one per target, trained on the same compounds and split.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByTestId("scorecard")).toHaveTextContent("solubility");
  });

  it("says one model learned every target when the training was joint", () => {
    render(<Scorecards scorecards={[card("a", true), card("b", true)]} />);

    expect(
      screen.getByText(
        "One model learned all 2 targets jointly. Each tab scores it on one target.",
      ),
    ).toBeInTheDocument();
  });

  it("renders a single target as a bare scorecard, without tabs", () => {
    render(<Scorecards scorecards={[card("solubility", false)]} />);

    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.getByTestId("scorecard")).toHaveTextContent("solubility");
  });
});
