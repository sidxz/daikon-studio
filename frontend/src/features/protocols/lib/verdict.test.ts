import type { Engine } from "@/features/engines";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import {
  computeOptimismGap,
  computeVerdict,
  describeBaseline,
  higherIsBetter,
  signed,
} from "./verdict";

function scorecard(overrides: Partial<ScorecardResponse>): ScorecardResponse {
  return {
    primary_metric: "r2",
    primary_metric_ci: null,
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

describe("describeBaseline with the engine manifest at hand", () => {
  const chemprop = {
    id: "chemprop-dmpnn",
    name: "Chemprop D-MPNN",
    conditions: [
      { key: "ensemble_size", label: "Ensemble size", options: [], option_labels: null },
      {
        key: "positive_weighting",
        label: "Positive-class weighting",
        options: ["none", "balanced"],
        option_labels: ["None", "Balanced"],
      },
      {
        key: "rdkit_descriptors",
        label: "Add RDKit descriptors",
        options: [],
        option_labels: null,
      },
    ],
  } as unknown as Engine;

  it("names what differs by the setting's own label, as the form shows it", () => {
    expect(
      describeBaseline(
        {
          baseline_engine_id: "chemprop-dmpnn",
          engine_id: "chemprop-dmpnn",
          conditions: { ensemble_size: 3, positive_weighting: "none", rdkit_descriptors: true },
          baseline_conditions: {
            ensemble_size: 1,
            positive_weighting: "balanced",
            rdkit_descriptors: false,
          },
        },
        [chemprop],
      ),
    ).toBe(
      "the same engine with Ensemble size 1, Positive-class weighting Balanced, Add RDKit descriptors off",
    );
  });

  it("names a different baseline engine by its name, not its id", () => {
    expect(
      describeBaseline(
        {
          baseline_engine_id: "chemprop-dmpnn",
          engine_id: "ecfp4-xgboost",
          conditions: {},
          baseline_conditions: {},
        },
        [chemprop],
      ),
    ).toBe("Chemprop D-MPNN");
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

  it("names the engine, not 'undefined', for a legacy blob with no recorded baseline conditions", () => {
    // A Run trained before the baseline became choosable: same engine on both
    // sides, non-default conditions, and `baseline_conditions == {}` because
    // the field predates this branch. Diffing against an empty object would
    // render every one of the model's own keys as "key = undefined".
    const described = describeBaseline({
      baseline_engine_id: "ecfp4-randomforest",
      engine_id: "ecfp4-randomforest",
      conditions: { n_estimators: 800 },
      baseline_conditions: {},
    });
    expect(described).not.toContain("undefined");
    expect(described).toBe("ecfp4-randomforest");
  });

  it("names every key that differs, not just the first, when several settings diverge", () => {
    const described = describeBaseline({
      baseline_engine_id: "chemprop-dmpnn",
      engine_id: "chemprop-dmpnn",
      conditions: { pretrained: "CheMeleon", depth: 6, message_hidden_dim: 2048, epochs: 50 },
      baseline_conditions: { pretrained: "none", depth: 3, message_hidden_dim: 300, epochs: 50 },
    });
    expect(described).toContain("pretrained");
    expect(described).toContain("depth");
    expect(described).toContain("message_hidden_dim");
    expect(described).not.toContain("epochs");
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
  });

  it("lets the backend's reason beat the vocabulary when the comparison was switched off", () => {
    // The one string whose job is to reach the reader before the per-strategy
    // sentence does. Nothing pinned it before, and "no comparison was recorded"
    // reads as a missing run rather than as a choice someone made.
    const gap = computeOptimismGap(
      scorecard({
        split_strategy: "scaffold",
        random_split_metrics: null,
        random_split_unavailable: "The random-split comparison was switched off for this run.",
      }),
    );
    expect(gap.message).toContain("switched off");
  });

  it("does not tell a predefined split it was scored on a random one", () => {
    // The old copy said "Not applicable: the model was scored on a random split" for
    // every strategy whose vocabulary has no group. That is flatly false here: the
    // partitions came out of the user's file.
    const gap = computeOptimismGap(
      scorecard({ split_strategy: "predefined", random_split_metrics: null }),
    );
    expect(gap.message).not.toContain("random split");
    expect(gap.message).toContain("from your file");
    expect(
      computeOptimismGap(scorecard({ random_split_unavailable: "the split had one class" })).kind,
    ).toBe("unavailable");
  });

  it("measures the gap on a sequence split too", () => {
    // Every grouped split gets the random-split comparison, not just scaffold.
    const gap = computeOptimismGap(
      scorecard({
        split_strategy: "identity",
        primary_metric: "r2",
        metrics: { r2: 0.389 },
        random_split_metrics: { r2: 0.758 },
      }),
    );
    expect(gap.kind).toBe("shown");
    expect(gap.held).toBeCloseTo(0.389);
    expect(gap.gap).toBeCloseTo(0.369);
  });

  it("never blames a missing comparison on a random split that did not happen", () => {
    const gap = computeOptimismGap(scorecard({ split_strategy: "position" }));
    expect(gap.kind).toBe("unavailable");
    expect(gap.message).not.toMatch(/scored on a random split/);
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

describe("a margin inside the bootstrap interval", () => {
  it("is within noise when the baseline sits inside the bootstrap interval", () => {
    const v = computeVerdict(
      scorecard({
        metrics: { r2: 0.7 },
        baseline_metrics: { r2: 0.6 },
        primary_metric_ci: [0.5, 0.8],
      }),
    );
    expect(v.kind).toBe("within-noise");
    expect(v.ci).toEqual([0.5, 0.8]);
  });

  it("still beats when the baseline is outside the interval", () => {
    expect(computeVerdict(scorecard({ primary_metric_ci: [0.65, 0.75] })).kind).toBe("beats");
  });

  it("carries no noise floor when the interval, not the assay noise, decides", () => {
    // The band words the two reasons differently, and an R² margin held against
    // a noise floor in the readout's units would be arithmetic on unrelated
    // quantities.
    const v = computeVerdict(scorecard({ primary_metric_ci: [0.5, 0.8], noise_floor: 0.151 }));
    expect(v.kind).toBe("within-noise");
    expect(v.noiseFloor ?? null).toBeNull();
  });
});

describe("a paired interval on the difference", () => {
  const rmse = (over: Partial<ScorecardResponse>) =>
    scorecard({
      primary_metric: "rmse",
      metrics: { rmse: 3.03 },
      baseline_metrics: { rmse: 3.19 },
      // The model's own interval holds the baseline: the old check would say noise.
      primary_metric_ci: [2.65, 3.34],
      ...over,
    });

  it("beats when the whole interval favors the model, whatever the old check says", () => {
    const v = computeVerdict(rmse({ difference_ci: [-0.176, -0.139] }));
    expect(v.kind).toBe("beats");
    expect(v.difference).toEqual([-0.176, -0.139]);
  });

  it("is within noise when the interval reaches zero, whatever the old check says", () => {
    const v = computeVerdict(
      scorecard({ primary_metric_ci: [0.65, 0.75], difference_ci: [-0.02, 0.2] }),
    );
    expect(v.kind).toBe("within-noise");
    expect(v.noiseFloor ?? null).toBeNull();
  });

  it("is within noise when the interval sits on the baseline's side of a model lead", () => {
    expect(computeVerdict(scorecard({ difference_ci: [-0.2, -0.01] })).kind).toBe("within-noise");
  });

  it("lets the assay noise floor speak first", () => {
    const v = computeVerdict(rmse({ difference_ci: [-0.176, -0.139], noise_floor: 0.3 }));
    expect(v.kind).toBe("within-noise");
    expect(v.noiseFloor).toBe(0.3);
  });

  it("leaves a loss a loss", () => {
    const v = computeVerdict(scorecard({ metrics: { r2: 0.5 }, difference_ci: [-0.2, -0.05] }));
    expect(v.kind).toBe("no-better");
  });

  it("keeps the old check for a card that has no paired interval", () => {
    const v = computeVerdict(rmse({}));
    expect(v.kind).toBe("within-noise");
    expect(v.difference ?? null).toBeNull();
  });
});

describe("signed", () => {
  it("marks a gain and leaves a loss its own sign", () => {
    expect(signed(0.2)).toBe("+0.200");
    expect(signed(-0.08)).toBe("-0.080");
    expect(signed(0)).toBe("0.000");
  });
});
