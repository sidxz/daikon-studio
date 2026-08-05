import { describe, expect, it } from "vitest";
import type { SweepRun } from "../types";
import { baselineDelta, formatMetric, rankRuns } from "./rank";

function run(id: string, metrics: SweepRun["metrics"], status = "ready"): SweepRun {
  return { id, status, metrics } as SweepRun;
}

describe("rankRuns", () => {
  it("ranks a higher MCC first", () => {
    const ranked = rankRuns([
      run("a", { primary_metric: "mcc", value: 0.4, baseline_value: 0.3 }),
      run("b", { primary_metric: "mcc", value: 0.7, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["b", "a"]);
  });

  it("ranks a lower RMSE first", () => {
    const ranked = rankRuns([
      run("a", { primary_metric: "rmse", value: 0.9, baseline_value: 1.1 }),
      run("b", { primary_metric: "rmse", value: 0.4, baseline_value: 1.1 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["b", "a"]);
  });

  it("puts unfinished runs last rather than treating them as zero", () => {
    const ranked = rankRuns([
      run("pending", null, "running"),
      run("scored", { primary_metric: "mcc", value: 0.1, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["scored", "pending"]);
  });

  it("puts a run whose metric is undefined last, alongside the unfinished", () => {
    const ranked = rankRuns([
      run("undefined-metric", { primary_metric: "mcc", value: null, baseline_value: 0.3 }),
      run("scored", { primary_metric: "mcc", value: 0.1, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["scored", "undefined-metric"]);
  });

  it("keeps submission order among equally unrankable runs", () => {
    const ranked = rankRuns([run("first", null, "pending"), run("second", null, "pending")]);
    expect(ranked.map((r) => r.id)).toEqual(["first", "second"]);
  });
});

describe("formatMetric", () => {
  it("renders the metric name and value", () => {
    expect(formatMetric({ primary_metric: "mcc", value: 0.4123, baseline_value: 0.3 })).toBe(
      "MCC 0.412",
    );
  });

  it("falls back to a dash when the value is missing", () => {
    expect(formatMetric(null)).toBe("—");
    expect(formatMetric({ primary_metric: "mcc", value: null, baseline_value: 0.3 })).toBe("—");
  });
});

describe("baselineDelta", () => {
  it("is positive when a higher-is-better metric beats its baseline", () => {
    expect(baselineDelta({ primary_metric: "mcc", value: 0.7, baseline_value: 0.3 })).toBeCloseTo(
      0.4,
    );
  });

  it("is negative when a higher-is-better metric loses to its baseline", () => {
    expect(baselineDelta({ primary_metric: "mcc", value: 0.2, baseline_value: 0.3 })).toBeCloseTo(
      -0.1,
    );
  });

  it("is positive when a lower-is-better metric beats its baseline", () => {
    expect(baselineDelta({ primary_metric: "rmse", value: 0.4, baseline_value: 1.1 })).toBeCloseTo(
      0.7,
    );
  });

  it("is negative when a lower-is-better metric loses to its baseline", () => {
    expect(baselineDelta({ primary_metric: "rmse", value: 1.3, baseline_value: 1.1 })).toBeCloseTo(
      -0.2,
    );
  });

  it("is null when either side is missing", () => {
    expect(baselineDelta(null)).toBeNull();
    expect(baselineDelta({ primary_metric: "mcc", value: null, baseline_value: 0.3 })).toBeNull();
    expect(baselineDelta({ primary_metric: "mcc", value: 0.4, baseline_value: null })).toBeNull();
  });
});
