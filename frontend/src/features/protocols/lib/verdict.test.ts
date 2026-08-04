import type { ScorecardResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import { computeOptimismGap, computeVerdict, describeBaseline, higherIsBetter } from "./verdict";

function scorecard(overrides: Partial<ScorecardResponse>): ScorecardResponse {
  return {
    primary_metric: "r2",
    prediction_kind: "numeric",
    metrics: { r2: 0.7 },
    metrics_undefined: null,
    baseline_engine_id: "ecfp4-randomforest",
    baseline_metrics: { r2: 0.6 },
    baseline_is_self: false,
    random_split_metrics: null,
    random_split_unavailable: null,
    random_split_metrics_undefined: null,
    noise_floor: null,
    worst_rows: [],
    applicability_coverage: null,
    unit: null,
    direction: null,
    split_strategy: "scaffold",
    ...overrides,
  } as ScorecardResponse;
}

describe("verdict", () => {
  it("says a higher R² beats the baseline", () => {
    expect(computeVerdict(scorecard({})).kind).toBe("beats");
  });

  it("says a lower R² does not", () => {
    expect(computeVerdict(scorecard({ metrics: { r2: 0.5 } })).kind).toBe("no-better");
  });

  it("inverts for error metrics, where smaller is the better model", () => {
    // The readout's own direction is irrelevant here: a low RMSE is a good
    // model even when a high IC50 is the desirable measurement.
    const card = scorecard({
      primary_metric: "rmse",
      metrics: { rmse: 0.4 },
      baseline_metrics: { rmse: 0.9 },
    });
    expect(higherIsBetter("rmse")).toBe(false);
    expect(computeVerdict(card).kind).toBe("beats");
  });

  it("renders no comparison when the model IS the baseline", () => {
    const verdict = computeVerdict(scorecard({ baseline_is_self: true }));
    expect(verdict.kind).toBe("is-baseline");
    // A comparison that never happened must not be shown as though it had.
    expect(verdict.baseline).toBeNull();
    expect(verdict.delta).toBeNull();
  });

  it("refuses to guess when a metric is missing", () => {
    expect(computeVerdict(scorecard({ metrics: {} })).kind).toBe("unknown");
  });
});

describe("describeBaseline", () => {
  it("describes a same-engine comparison by what differs, not by the engine id", () => {
    // Both sides are chemprop-dmpnn. Naming only the engine would render
    // "chemprop-dmpnn versus chemprop-dmpnn", which explains nothing.
    expect(
      describeBaseline({
        baseline_engine_id: "chemprop-dmpnn",
        engine_id: "chemprop-dmpnn",
        conditions: { pretrained: "CheMeleon", epochs: 50 },
        baseline_conditions: { pretrained: "none", epochs: 50 },
      }),
    ).toContain("pretrained");
  });

  it("names the engine when the two sides are different engines", () => {
    expect(
      describeBaseline({
        baseline_engine_id: "ecfp4-randomforest",
        engine_id: "chemprop-dmpnn",
        conditions: {},
        baseline_conditions: {},
      }),
    ).toContain("ecfp4-randomforest");
  });
});

describe("optimism gap", () => {
  it("reports how much the random split flattered the model", () => {
    const gap = computeOptimismGap(
      scorecard({ metrics: { r2: 0.6 }, random_split_metrics: { r2: 0.85 } }),
    );
    expect(gap.kind).toBe("shown");
    expect(gap.gap).toBeCloseTo(0.25);
  });

  it("keeps the gap positive for error metrics too", () => {
    const gap = computeOptimismGap(
      scorecard({
        primary_metric: "rmse",
        metrics: { rmse: 1.2 },
        random_split_metrics: { rmse: 0.8 },
      }),
    );
    expect(gap.gap).toBeCloseTo(0.4);
  });

  it("distinguishes 'not applicable' from 'could not be computed'", () => {
    // Two nulls mean the question does not arise. A message means it was tried
    // and failed. These must never render the same way.
    expect(computeOptimismGap(scorecard({ split_strategy: "random" })).kind).toBe("not-applicable");
    expect(
      computeOptimismGap(scorecard({ random_split_unavailable: "the split had one class" })).kind,
    ).toBe("unavailable");
  });
});

describe("a margin smaller than the assay noise", () => {
  const narrow = {
    primary_metric: "rmse",
    metrics: { rmse: 1.2 },
    baseline_metrics: { rmse: 1.228 },
  };

  it("refuses to call a sub-noise win a win", () => {
    // The exact case ESOL produced live: 0.028 better on RMSE, against
    // duplicate measurements that disagree by 0.151.
    const verdict = computeVerdict(scorecard({ ...narrow, noise_floor: 0.151 }));
    expect(verdict.kind).toBe("within-noise");
    expect(verdict.noiseFloor).toBe(0.151);
  });

  it("still calls a win a win when it clears the noise", () => {
    const verdict = computeVerdict(
      scorecard({
        primary_metric: "rmse",
        metrics: { rmse: 0.6 },
        baseline_metrics: { rmse: 1.228 },
        noise_floor: 0.151,
      }),
    );
    expect(verdict.kind).toBe("beats");
  });

  it("does not hold a dimensionless metric against a noise floor in the readout's units", () => {
    // R² has no units; comparing it to a spread in log mol/L would be
    // arithmetic on unrelated quantities.
    const verdict = computeVerdict(
      scorecard({ metrics: { r2: 0.687 }, baseline_metrics: { r2: 0.672 }, noise_floor: 0.151 }),
    );
    expect(verdict.kind).toBe("beats");
  });

  it("leaves a loss a loss regardless of noise", () => {
    const verdict = computeVerdict(
      scorecard({
        primary_metric: "rmse",
        metrics: { rmse: 1.3 },
        baseline_metrics: { rmse: 1.228 },
        noise_floor: 0.151,
      }),
    );
    expect(verdict.kind).toBe("no-better");
  });
});
