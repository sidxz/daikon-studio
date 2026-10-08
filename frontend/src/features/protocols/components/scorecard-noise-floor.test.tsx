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
    ...over,
  }) as unknown as ScorecardResponse;

describe("the noise floor when there is none", () => {
  it("says why it is missing when deduplication was switched off", () => {
    // Otherwise the panel simply vanishes, which reads as a bug rather than as the
    // consequence of a choice the scientist made on purpose.
    render(<ScorecardView scorecard={card({ deduplicated: false })} />);
    expect(screen.getByText(/deduplication was switched off/)).toBeInTheDocument();
  });

  it("stays silent for a target that simply has no replicates, as before", () => {
    render(<ScorecardView scorecard={card({ deduplicated: true })} />);
    expect(screen.queryByText(/repeated measurements/i)).not.toBeInTheDocument();
  });

  it("stays silent for a scorecard written before the toggle existed", () => {
    render(<ScorecardView scorecard={card({ deduplicated: null })} />);
    expect(screen.queryByText(/repeated measurements/i)).not.toBeInTheDocument();
  });
});
