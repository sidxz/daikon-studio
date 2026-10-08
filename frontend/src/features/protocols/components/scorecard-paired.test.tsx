import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { computeVerdict } from "../lib/verdict";
import { ScorecardView, bootstrapData } from "./scorecard-view";

vi.mock("@/features/engines", () => ({ useEngines: () => ({ data: [] }) }));
vi.mock("./scorecard-diagnostics", () => ({
  ScorecardDiagnostics: () => null,
  SplitComparison: () => null,
}));
vi.mock("./largest-errors", () => ({ LargestErrors: () => null }));
vi.mock("@/shared/components/explainers/figures/bootstrap", () => ({
  BootstrapExplainer: () => null,
}));

const redraws = { edges: [0, 1, 2], counts: [500, 500] };
const card = (over: Partial<ScorecardResponse>) =>
  ({
    engine_id: "tanimoto-gp",
    baseline_engine_id: "ecfp4-randomforest",
    primary_metric: "rmse",
    prediction_kind: "value",
    metrics: { rmse: 3.03 },
    baseline_metrics: { rmse: 3.19 },
    primary_metric_ci: [2.65, 3.34],
    primary_metric_bootstrap: redraws,
    conditions: {},
    baseline_conditions: {},
    split_strategy: "scaffold",
    baseline_is_self: false,
    cutoff: null,
    baseline_cutoff: null,
    cutoff_note: null,
    noise_floor: null,
    deduplicated: true,
    parity: [],
    ...over,
  }) as unknown as ScorecardResponse;

describe("the regression band with a paired interval", () => {
  it("states the interval for the difference in place of the model's own range", () => {
    render(
      <ScorecardView
        scorecard={card({ difference_ci: [-0.176, -0.139], difference_bootstrap: redraws })}
      />,
    );
    expect(screen.getByText(/95% interval for the difference/)).toHaveTextContent(
      "-0.176 to -0.139",
    );
    expect(screen.queryByText(/Likely range for this/)).not.toBeInTheDocument();
    expect(screen.getByText("Outperforms the baseline")).toBeInTheDocument();
  });

  it("says the interval reaches zero when that is why the lead is uncertain", () => {
    render(
      <ScorecardView
        scorecard={card({ difference_ci: [-0.3, 0.05], difference_bootstrap: redraws })}
      />,
    );
    expect(screen.getByText(/The interval for the difference includes zero/)).toBeInTheDocument();
  });

  it("renders an older card exactly as before", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.getByText(/Likely range for this/)).toBeInTheDocument();
    expect(screen.getByText(/lies within this interval/)).toBeInTheDocument();
    expect(screen.queryByText(/interval for the difference/)).not.toBeInTheDocument();
  });
});

describe("the figure's data", () => {
  it("draws the difference against zero when the card is paired", () => {
    const paired = card({ difference_ci: [-0.176, -0.139], difference_bootstrap: redraws });
    const data = bootstrapData(paired, computeVerdict(paired));
    expect(data?.mode).toBe("difference");
    expect(data?.baseline).toBe(0);
    expect(data?.interval).toEqual([-0.176, -0.139]);
  });

  it("draws the model's own score against the baseline's on an older card", () => {
    const old = card({});
    const data = bootstrapData(old, computeVerdict(old));
    expect(data?.mode).toBe("score");
    expect(data?.baseline).toBe(3.19);
  });
});
