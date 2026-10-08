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

describe("the band when the split's own spread decides the verdict", () => {
  // The spread across draws is normally *larger* than resampling one test set, so
  // "the spread demotes a lead the paired interval endorsed" is this feature's common
  // case. The interval printed on the card then excludes zero, and the explanation
  // must not claim otherwise.
  const demoted = card({
    difference_ci: [-0.176, -0.139],
    difference_bootstrap: redraws,
    replicate_summary: { rmse: { mean: 3.05, sd: 0.1, n: 5 } },
    replicate_seeds: [8, 9, 10, 11, 12],
  });

  it("does not claim the difference includes zero when it plainly does not", () => {
    render(<ScorecardView scorecard={demoted} />);

    expect(screen.getByText(/95% interval for the difference/)).toHaveTextContent(
      "-0.176 to -0.139",
    );
    expect(
      screen.queryByText(/The interval for the difference includes zero/),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/lies within this interval/)).not.toBeInTheDocument();
  });

  it("explains the verdict it actually reached, and shows the number behind it", () => {
    render(<ScorecardView scorecard={demoted} />);

    expect(
      screen.getByText("Ahead of the baseline, but by less than the split itself moves the score"),
    ).toBeInTheDocument();
    // The deciding number must appear. Naming a reason and withholding its magnitude
    // leaves the reader no way to judge it.
    expect(screen.getByText(/0\.100/)).toBeInTheDocument();
  });

  it("does not claim the baseline sits inside the model's own interval either", () => {
    // The unpaired fallback is wrong in the same way: here the baseline's 3.19 is
    // outside [2.65, 3.34]... and even when it is inside, that is not why this
    // verdict was reached.
    render(
      <ScorecardView
        scorecard={card({
          replicate_summary: { rmse: { mean: 3.05, sd: 0.1, n: 5 } },
          replicate_seeds: [8, 9, 10, 11, 12],
        })}
      />,
    );

    expect(screen.queryByText(/lies within this interval/)).not.toBeInTheDocument();
    expect(screen.getByText(/0\.100/)).toBeInTheDocument();
  });

  it("still explains an assay-noise verdict the old way", () => {
    render(
      <ScorecardView
        scorecard={card({
          noise_floor: 0.5,
          replicate_summary: { rmse: { mean: 3.05, sd: 0.1, n: 5 } },
          replicate_seeds: [8, 9],
        })}
      />,
    );

    expect(screen.getByText(/assay noise floor/)).toBeInTheDocument();
  });

  it("renders a card with no draws exactly as before", () => {
    render(<ScorecardView scorecard={card({ difference_ci: [-0.3, 0.05] })} />);

    expect(screen.getByText(/The interval for the difference includes zero/)).toBeInTheDocument();
  });
});
