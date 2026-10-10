import type { ScorecardResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import { type Aggregate, aggregateTargets, averageableMetrics } from "./aggregate";

function card(overrides: Partial<ScorecardResponse>): ScorecardResponse {
  return {
    target: "t",
    joint_model: true,
    primary_metric: "auroc",
    metrics: { auroc: 0.8 },
    metrics_undefined: null,
    baseline_metrics: null,
    unit: null,
    ...overrides,
  } as ScorecardResponse;
}

/** One target scored on the shared metric, named and valued. */
const scored = (target: string, value: number, rest: Partial<ScorecardResponse> = {}) =>
  card({ target, metrics: { auroc: value }, ...rest });

/**
 * The average, or a failure naming the refusal. Averages are compared with
 * `toBeCloseTo` rather than matched: `(0.8 + 0.9) / 2` is 0.8500000000000001.
 */
function shown(result: Aggregate) {
  if (result.kind !== "shown") throw new Error(`expected an average, got: ${result.message}`);
  return result;
}

describe("aggregateTargets", () => {
  it("averages the primary metric over the targets that were scored", () => {
    const result = shown(aggregateTargets([scored("NR-AR", 0.8), scored("SR-MMP", 0.9)]));
    expect(result).toMatchObject({ metric: "auroc", unit: null, skipped: [] });
    expect(result.mean).toBeCloseTo(0.85, 12);
  });

  it("counts every target equally, so twelve endpoints average as twelve", () => {
    const values = [0.73, 0.81, 0.69, 0.9, 0.85, 0.77, 0.82, 0.88, 0.76, 0.79, 0.84, 0.71];
    const result = shown(aggregateTargets(values.map((value, i) => scored(`e${i}`, value))));
    expect(result.scored).toHaveLength(12);
    expect(result.mean).toBeCloseTo(values.reduce((a, b) => a + b, 0) / 12, 12);
  });

  it("refuses to average targets scored on different metrics, and names them", () => {
    const result = aggregateTargets([
      scored("NR-AR", 0.8),
      card({ target: "logS", primary_metric: "mae", metrics: { mae: 0.6 } }),
    ]);
    expect(result.kind).toBe("unavailable");
    if (result.kind !== "unavailable") return;
    expect(result.message).toContain("ROC AUC");
    expect(result.message).toContain("MAE");
  });

  it("refuses to average an error metric measured in different units", () => {
    const result = aggregateTargets([
      card({ target: "a", primary_metric: "mae", metrics: { mae: 0.4 }, unit: "nM" }),
      card({ target: "b", primary_metric: "mae", metrics: { mae: 0.6 }, unit: "log M" }),
    ]);
    expect(result.kind).toBe("unavailable");
    if (result.kind !== "unavailable") return;
    expect(result.message).toContain("nM");
    expect(result.message).toContain("log M");
  });

  it("averages an error metric when every target shares one unit, and reports it", () => {
    const result = shown(
      aggregateTargets([
        card({ target: "a", primary_metric: "mae", metrics: { mae: 0.4 }, unit: "log M" }),
        card({ target: "b", primary_metric: "mae", metrics: { mae: 0.6 }, unit: "log M" }),
      ]),
    );
    expect(result.unit).toBe("log M");
    expect(result.mean).toBeCloseTo(0.5, 12);
  });

  it("ignores a target's unit when the metric does not carry one", () => {
    // A binary endpoint cut from a micromolar readout keeps that readout's unit
    // on its scorecard, but ROC AUC is a ranking, not a concentration.
    const result = shown(
      aggregateTargets([scored("a", 0.8, { unit: "uM" }), scored("b", 0.9, { unit: "nM" })]),
    );
    expect(result.unit).toBeNull();
    expect(result.mean).toBeCloseTo(0.85, 12);
  });

  it("leaves an unscored target out of the average and says why", () => {
    const result = shown(
      aggregateTargets([
        scored("NR-AR", 0.8),
        scored("SR-MMP", 0.9),
        card({
          target: "NR-AhR",
          metrics: { auroc: null },
          metrics_undefined: { auroc: "no compound in the test set carries a label" },
        }),
      ]),
    );
    expect(result.mean).toBeCloseTo(0.85, 12);
    expect(result.scored.map((entry) => entry.target)).toEqual(["NR-AR", "SR-MMP"]);
    expect(result.skipped).toEqual([
      { target: "NR-AhR", reason: "no compound in the test set carries a label" },
    ]);
  });

  it("explains an unscored target the backend gave no reason for", () => {
    const result = shown(
      aggregateTargets([
        scored("a", 0.8),
        scored("b", 0.9),
        card({ target: "c", metrics: { auroc: null } }),
      ]),
    );
    expect(result.skipped[0]).toMatchObject({ target: "c" });
    expect(result.skipped[0].reason).toContain("ROC AUC");
  });

  it("refuses when only one target was scored", () => {
    const result = aggregateTargets([
      scored("a", 0.8),
      card({ target: "b", metrics: { auroc: null } }),
    ]);
    expect(result.kind).toBe("unavailable");
    if (result.kind !== "unavailable") return;
    expect(result.message).toContain("one target");
  });

  it("refuses on a single target, where there is nothing to average across", () => {
    expect(aggregateTargets([scored("a", 0.8)]).kind).toBe("unavailable");
    expect(aggregateTargets([]).kind).toBe("unavailable");
  });

  it("averages the comparison model over the same targets", () => {
    const result = shown(
      aggregateTargets([
        scored("a", 0.8, { baseline_metrics: { auroc: 0.7 } }),
        scored("b", 0.9, { baseline_metrics: { auroc: 0.75 } }),
      ]),
    );
    expect(result.mean).toBeCloseTo(0.85, 12);
    expect(result.baselineMean).toBeCloseTo(0.725, 12);
  });

  it("withholds the comparison average unless every scored target has one", () => {
    const result = shown(
      aggregateTargets([scored("a", 0.8, { baseline_metrics: { auroc: 0.7 } }), scored("b", 0.9)]),
    );
    expect(result.baselineMean).toBeNull();
  });

  it("ignores the comparison model on a target left out of the average", () => {
    // Averaging the model over two targets and the baseline over three would
    // print a difference that was never measured on the same endpoints.
    const result = shown(
      aggregateTargets([
        scored("a", 0.8, { baseline_metrics: { auroc: 0.7 } }),
        scored("b", 0.9, { baseline_metrics: { auroc: 0.8 } }),
        card({ target: "c", metrics: { auroc: null }, baseline_metrics: { auroc: 0.1 } }),
      ]),
    );
    expect(result.mean).toBeCloseTo(0.85, 12);
    expect(result.baselineMean).toBeCloseTo(0.75, 12);
  });

  it("puts the weakest target first when a higher score is better", () => {
    const result = shown(
      aggregateTargets([scored("good", 0.9), scored("bad", 0.5), scored("mid", 0.7)]),
    );
    expect(result.scored.map((entry) => entry.target)).toEqual(["bad", "mid", "good"]);
  });

  it("puts the weakest target first when a lower score is better", () => {
    const mae = (target: string, value: number) =>
      card({ target, primary_metric: "mae", metrics: { mae: value }, unit: "log M" });
    const result = shown(aggregateTargets([mae("good", 0.2), mae("bad", 1.4), mae("mid", 0.8)]));
    expect(result.scored.map((entry) => entry.target)).toEqual(["bad", "mid", "good"]);
  });
});

describe("choosing which metric to average", () => {
  // The protocol this was built for scores MCC as its primary metric while the
  // benchmark it reproduces is published as a mean ROC AUC. Averaging only the
  // primary metric leaves that comparison exactly where it was: in twelve tabs.
  const both = (target: string, mcc: number, auroc: number) =>
    card({ target, primary_metric: "mcc", metrics: { mcc, auroc } });

  it("averages a metric other than the primary one when asked", () => {
    const result = shown(aggregateTargets([both("a", 0.2, 0.8), both("b", 0.4, 0.9)], "auroc"));
    expect(result.metric).toBe("auroc");
    expect(result.mean).toBeCloseTo(0.85, 12);
  });

  it("still averages the primary metric by default", () => {
    const result = shown(aggregateTargets([both("a", 0.2, 0.8), both("b", 0.4, 0.9)]));
    expect(result.metric).toBe("mcc");
    expect(result.mean).toBeCloseTo(0.3, 12);
  });

  it("averages a chosen metric even where the targets' primary metrics differ", () => {
    // Disagreeing primary metrics block a default average, but a metric every
    // target carries is still a fair thing to average.
    const result = shown(
      aggregateTargets(
        [
          card({ target: "a", primary_metric: "mcc", metrics: { mcc: 0.2, auroc: 0.8 } }),
          card({ target: "b", primary_metric: "auprc", metrics: { auprc: 0.4, auroc: 0.9 } }),
        ],
        "auroc",
      ),
    );
    expect(result.mean).toBeCloseTo(0.85, 12);
  });

  it("orders the choices with the primary metric first", () => {
    expect(averageableMetrics([both("a", 0.2, 0.8), both("b", 0.4, 0.9)])).toEqual([
      "mcc",
      "auroc",
    ]);
  });

  it("offers only metrics at least two targets actually carry", () => {
    const metrics = averageableMetrics([
      card({ target: "a", primary_metric: "mcc", metrics: { mcc: 0.2, auroc: 0.8, r2: 0.1 } }),
      card({ target: "b", primary_metric: "mcc", metrics: { mcc: 0.4, auroc: 0.9, r2: null } }),
    ]);
    expect(metrics).toEqual(["mcc", "auroc"]);
  });

  it("offers nothing to choose between when there is one target", () => {
    expect(averageableMetrics([card({ target: "a", metrics: { auroc: 0.8 } })])).toEqual([]);
  });
});

/**
 * The twelve endpoints of the Tox21 case study, at the ROC AUC this product
 * measured for each, for three configurations trained through the wizard.
 *
 * These are the per-target scores of our own models, not Tox21's labels. The
 * means on the right were computed in Python while writing that case study up,
 * before this module existed -- which is the point of pinning them here. The
 * hand-computed mean is the reason this function is being written at all, so it
 * is the one oracle available that this code had no hand in producing.
 */
const TOX21 = {
  endpoints: [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
  ],
  runs: [
    {
      name: "Chemprop D-MPNN",
      auroc: [0.903, 0.836, 0.915, 0.813, 0.789, 0.738, 0.798, 0.804, 0.804, 0.804, 0.922, 0.839],
      mean: 0.83,
    },
    {
      name: "RF + RDKit descriptors",
      auroc: [0.91, 0.75, 0.741, 0.802, 0.796, 0.782, 0.715, 0.773, 0.77, 0.808, 0.948, 0.813],
      mean: 0.801,
    },
    {
      name: "RF, ECFP4 only",
      auroc: [0.872, 0.785, 0.762, 0.746, 0.735, 0.699, 0.72, 0.71, 0.682, 0.747, 0.879, 0.783],
      mean: 0.76,
    },
  ],
};

describe("aggregateTargets on the Tox21 case study", () => {
  for (const run of TOX21.runs) {
    it(`reproduces the mean ROC AUC computed by hand for ${run.name}`, () => {
      const result = shown(
        aggregateTargets(
          TOX21.endpoints.map((endpoint, index) =>
            scored(endpoint, run.auroc[index], { joint_model: true }),
          ),
        ),
      );
      expect(result.scored).toHaveLength(12);
      expect(result.skipped).toEqual([]);
      expect(Number(result.mean.toFixed(3))).toBe(run.mean);
    });
  }

  it("names the weakest endpoint of the reproducing run", () => {
    const chemprop = TOX21.runs[0];
    const result = shown(
      aggregateTargets(
        TOX21.endpoints.map((endpoint, index) => scored(endpoint, chemprop.auroc[index])),
      ),
    );
    // The number to act on: the mean of 0.830 hides a 0.738 endpoint, which is
    // the whole reason the per-target list is sorted rather than alphabetical.
    expect(result.scored[0]).toMatchObject({ target: "NR-ER-LBD", value: 0.738 });
    expect(result.scored.at(-1)).toMatchObject({ target: "SR-MMP", value: 0.922 });
  });
});
