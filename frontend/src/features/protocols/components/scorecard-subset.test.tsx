import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScorecardView } from "./scorecard-view";

vi.mock("@/features/engines", () => ({ useEngines: () => ({ data: [] }) }));
vi.mock("./scorecard-diagnostics", () => ({
  ScorecardDiagnostics: () => null,
  SplitComparison: () => null,
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
    baseline_metrics: { rmse: 0.7 },
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

describe("a metric over a flagged part of the test set", () => {
  it("names the column and the denominator, so it cannot be read as the overall score", () => {
    render(
      <ScorecardView
        scorecard={card({
          subset_column: "cliff_mol",
          subset_count: 245,
          subset_total: 666,
          subset_metric: 0.71,
        })}
      />,
    );
    expect(screen.getByText(/cliff_mol/)).toBeInTheDocument();
    expect(screen.getByText(/245 of 666/)).toBeInTheDocument();
  });

  it("says nothing was flagged rather than showing an empty number", () => {
    render(
      <ScorecardView
        scorecard={card({
          subset_column: "cliff_mol",
          subset_count: 0,
          subset_total: 666,
          subset_metric: null,
        })}
      />,
    );
    expect(screen.getByText(/No test row/)).toBeInTheDocument();
  });

  it("does not say nothing was flagged when 245 rows were", () => {
    // The null metric has three causes and this is the second: a binary subset whose
    // flagged rows are all one class. Saying "no test row carries this flag" above a
    // count of 245 contradicts the line directly beneath it.
    render(
      <ScorecardView
        scorecard={card({
          primary_metric: "mcc",
          metrics: { mcc: 0.5 },
          subset_column: "cliff_mol",
          subset_count: 245,
          subset_total: 666,
          subset_metric: null,
        })}
      />,
    );
    expect(screen.queryByText(/No test row/)).not.toBeInTheDocument();
    expect(screen.getByText(/every flagged compound has the same label/)).toBeInTheDocument();
  });

  it("shows no section at all when no column was chosen", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.queryByText(/flagged by/)).not.toBeInTheDocument();
  });
});
