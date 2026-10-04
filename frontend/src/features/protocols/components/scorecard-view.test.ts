import type { ScorecardResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import type { Verdict } from "../lib/verdict";
import { bootstrapData } from "./scorecard-view";

const v = (kind: Verdict["kind"], ci: [number, number] | null): Verdict => ({
  kind,
  headline: "",
  model: 0.7,
  baseline: 0.6,
  delta: 0.1,
  ci,
});

const redraws = { edges: [0.5, 0.7, 0.9], counts: [400, 600] };
const card = (over: Partial<ScorecardResponse> = {}) =>
  ({
    primary_metric: "mcc",
    primary_metric_bootstrap: redraws,
    prediction_kind: "probability",
    cutoff: null,
    parity: [{ actual: 1, predicted: 0.8, similarity: null }],
    parity_sampled_from: 9000,
    ...over,
  }) as unknown as ScorecardResponse;

describe("bootstrapData", () => {
  it("shows for a comparison with an interval", () => {
    for (const kind of ["beats", "within-noise", "no-better"] as const)
      expect(bootstrapData(card(), v(kind, [0.6, 0.8]))).not.toBeNull();
  });
  it("hides without an interval, its redraws, or a comparison", () => {
    expect(bootstrapData(card(), v("beats", null))).toBeNull();
    expect(
      bootstrapData(card({ primary_metric_bootstrap: null }), v("beats", [0.6, 0.8])),
    ).toBeNull();
    expect(bootstrapData(card(), v("is-baseline", [0.6, 0.8]))).toBeNull();
    expect(bootstrapData(card(), v("unknown", [0.6, 0.8]))).toBeNull();
  });
  it("carries the card's own numbers, cutoff and full test size", () => {
    const data = bootstrapData(card({ cutoff: 0.31 }), v("beats", [0.6, 0.8]));
    expect(data).toMatchObject({
      higherIsBetter: true,
      interval: [0.6, 0.8],
      baseline: 0.6,
      redraws,
      testSize: 9000,
      cutoff: 0.31,
    });
  });
  it("has no cutoff for regression, and an untuned classifier sits at 0.5", () => {
    const regression = card({ primary_metric: "rmse", prediction_kind: "value" });
    expect(bootstrapData(regression, v("beats", [0.6, 0.8]))?.cutoff).toBeNull();
    expect(bootstrapData(regression, v("beats", [0.6, 0.8]))?.higherIsBetter).toBe(false);
    expect(bootstrapData(card(), v("beats", [0.6, 0.8]))?.cutoff).toBe(0.5);
  });
});
