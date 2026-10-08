import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScorecardView } from "./scorecard-view";

vi.mock("@/features/engines", () => ({ useEngines: () => ({ data: [] }) }));
vi.mock("./scorecard-diagnostics", () => ({
  ScorecardDiagnostics: () => null,
  SplitComparison: () => null,
  SplitDrawSpread: () => null,
}));
vi.mock("./largest-errors", () => ({ LargestErrors: () => null }));
vi.mock("@/shared/components/explainers/figures/bootstrap", () => ({
  BootstrapExplainer: () => null,
}));

const card = (over: Partial<ScorecardResponse>) =>
  ({
    engine_id: "tanimoto-gp",
    baseline_engine_id: "ecfp4-randomforest",
    primary_metric: "rmse",
    metrics: { rmse: 0.64 },
    baseline_metrics: null,
    conditions: {},
    baseline_conditions: {},
    split_strategy: "predefined",
    baseline_is_self: false,
    cutoff: null,
    baseline_cutoff: null,
    cutoff_note: null,
    noise_floor: null,
    deduplicated: true,
    subset_column: null,
    subset_count: null,
    subset_total: null,
    subset_metric: null,
    ...over,
  }) as unknown as ScorecardResponse;

describe("a scorecard for a run that fitted no baseline", () => {
  it("says the baseline was not fitted rather than that a metric was undefined", () => {
    render(<ScorecardView scorecard={card({})} />);
    // Said on the verdict band and again on the metrics section; both are correct.
    expect(screen.getAllByText(/No baseline was fitted/).length).toBeGreaterThan(0);
  });

  it("never claims a comparison model was evaluated on the same test set", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.queryByText(/evaluated on the same test set/)).not.toBeInTheDocument();
  });

  it("does not blame an undefined metric for the missing comparison", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.queryByText(/could not be computed/)).not.toBeInTheDocument();
  });

  it("still reports the model's own score", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.getByText("0.640")).toBeInTheDocument();
  });
});
