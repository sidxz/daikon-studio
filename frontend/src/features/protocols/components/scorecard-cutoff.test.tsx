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
  SplitDrawSpread: () => null,
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

  it("never rounds a cutoff below 1 up to 1", () => {
    render(<ScorecardView scorecard={card({ cutoff: 0.99997, baseline_cutoff: 0.9867 })} />);
    expect(
      screen.getByText(
        "At cutoff 0.99997, tuned on validation. Baseline at its own tuned cutoff 0.987.",
      ),
    ).toBeInTheDocument();
  });

  it("says the baseline stayed at 0.5 when only the model's cutoff could be tuned", () => {
    render(<ScorecardView scorecard={card({ cutoff: 0.031, baseline_cutoff: null })} />);
    expect(
      screen.getByText(
        "At cutoff 0.031, tuned on validation. Baseline at 0.5: its validation predictions could not support a cutoff.",
      ),
    ).toBeInTheDocument();
  });

  it("adds the baseline's cutoff to the reason the model was not tuned", () => {
    const note = "Not tuned: the validation set has 3 active and 40 inactive compounds for 'y'.";
    render(<ScorecardView scorecard={card({ cutoff_note: note, baseline_cutoff: 0.12 })} />);
    expect(screen.getByText(`${note} Baseline at its own tuned cutoff 0.12.`)).toBeInTheDocument();
  });
});

describe("the cutoff when there is no comparison to make", () => {
  it("shows the cutoff of a model that is its own baseline, without a baseline sentence", () => {
    render(
      <ScorecardView
        scorecard={card({ baseline_is_self: true, cutoff: 0.031, baseline_cutoff: 0.031 })}
      />,
    );
    expect(screen.getByText("At cutoff 0.031, tuned on validation.")).toBeInTheDocument();
    expect(screen.queryByText(/Baseline at/)).not.toBeInTheDocument();
  });

  it("shows why tuning did not happen on a model that is its own baseline", () => {
    const note = "Not tuned: the validation set has 6 active and 94 inactive compounds for 'y'.";
    render(<ScorecardView scorecard={card({ baseline_is_self: true, cutoff_note: note })} />);
    expect(screen.getByText(note)).toBeInTheDocument();
  });

  it("shows the cutoff when the metric could not be computed for a comparison", () => {
    render(
      <ScorecardView
        scorecard={card({ metrics: { mcc: null }, cutoff: 0.031, baseline_cutoff: 0.12 })}
      />,
    );
    expect(screen.getByText("Cannot be compared to the baseline")).toBeInTheDocument();
    expect(
      screen.getByText(
        "At cutoff 0.031, tuned on validation. Baseline at its own tuned cutoff 0.12.",
      ),
    ).toBeInTheDocument();
  });
});
