import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScorecardView } from "./scorecard-view";

// The band is what is under test; the charts, the engine lookup and the
// animated explainer are not.
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
    engine_id: "chemprop-dmpnn",
    baseline_engine_id: "ecfp4-randomforest",
    primary_metric: "mcc",
    metrics: { mcc: 0.5 },
    baseline_metrics: { mcc: 0.4 },
    conditions: {},
    baseline_conditions: {},
    split_strategy: "scaffold",
    baseline_is_self: false,
    cutoff: null,
    baseline_cutoff: null,
    cutoff_note: null,
    ...over,
  }) as unknown as ScorecardResponse;

describe("the cutoff on the verdict band", () => {
  it("shows the model's and the baseline's tuned cutoffs", () => {
    render(<ScorecardView scorecard={card({ cutoff: 0.031, baseline_cutoff: 0.12 })} />);
    expect(
      screen.getByText(
        "At cutoff 0.031, tuned on validation. Baseline at its own tuned cutoff 0.12.",
      ),
    ).toBeInTheDocument();
  });

  it("ends after the model's cutoff when the baseline could not be tuned", () => {
    render(<ScorecardView scorecard={card({ cutoff: 0.031, baseline_cutoff: null })} />);
    expect(screen.getByText("At cutoff 0.031, tuned on validation.")).toBeInTheDocument();
  });

  it("says why tuning did not happen instead of a cutoff", () => {
    const note = "Not tuned: the validation set has 3 active and 40 inactive compounds for 'y'.";
    render(<ScorecardView scorecard={card({ cutoff_note: note })} />);
    expect(screen.getByText(note)).toBeInTheDocument();
    expect(screen.queryByText(/tuned on validation/)).not.toBeInTheDocument();
  });

  it("says nothing when tuning was not requested", () => {
    render(<ScorecardView scorecard={card({})} />);
    expect(screen.queryByText(/cutoff/i)).not.toBeInTheDocument();
  });
});
