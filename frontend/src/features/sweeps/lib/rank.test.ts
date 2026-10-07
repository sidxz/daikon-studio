import { describe, expect, it } from "vitest";
import type { SweepRun } from "../types";
import {
  baselineDelta,
  formatMetric,
  headlineFor,
  sortDirection,
  sortRuns,
  sweepTargets,
} from "./rank";

const run = (id: string, targets: [string, string, number | null, number | null][]): SweepRun =>
  ({
    id,
    metrics: {
      targets: targets.map(([column, primary_metric, value, baseline_value]) => ({
        column,
        primary_metric,
        value,
        baseline_value,
      })),
    },
  }) as unknown as SweepRun;

const a = run("a", [
  ["reactive", "mcc", 0.2, 0.1],
  ["solubility", "rmse", 0.5, 0.7],
]);
const b = run("b", [
  ["reactive", "mcc", 0.4, 0.1],
  ["solubility", "rmse", 0.9, 0.7],
]);
const pending = { id: "p", metrics: null } as unknown as SweepRun;

describe("sweepTargets", () => {
  it("lists every target once, in the order the runs report them", () => {
    expect(sweepTargets([pending, a, b])).toEqual(["reactive", "solubility"]);
  });
});

describe("sortRuns", () => {
  it("keeps submission order until a target is chosen", () => {
    expect(sortRuns([a, b, pending], null).map((r) => r.id)).toEqual(["a", "b", "p"]);
  });

  it("ranks by the chosen target in that metric's direction, unmeasured last", () => {
    expect(sortRuns([pending, a, b], "reactive").map((r) => r.id)).toEqual(["b", "a", "p"]);
    expect(sortRuns([pending, a, b], "solubility").map((r) => r.id)).toEqual(["a", "b", "p"]);
  });

  it("puts a run whose metric is undefined last, alongside the unfinished", () => {
    const undefinedMetric = run("u", [["reactive", "mcc", null, 0.1]]);
    expect(sortRuns([undefinedMetric, a, pending], "reactive").map((r) => r.id)).toEqual([
      "a",
      "u",
      "p",
    ]);
  });
});

describe("sortDirection", () => {
  it("is ascending for an error metric and descending for a score", () => {
    expect(sortDirection([pending, a, b], "solubility")).toBe("ascending");
    expect(sortDirection([pending, a, b], "reactive")).toBe("descending");
  });
});

describe("formatMetric and baselineDelta", () => {
  it("read one target's headline, signed so positive is always better", () => {
    expect(formatMetric(headlineFor(a, "solubility"))).toBe("RMSE 0.500");
    expect(baselineDelta(headlineFor(a, "solubility"))).toBeCloseTo(0.2);
    expect(baselineDelta(headlineFor(b, "reactive"))).toBeCloseTo(0.3);
    expect(formatMetric(headlineFor(pending, "reactive"))).toBe("N/A");
  });

  it("gives no delta when either side is missing", () => {
    expect(
      baselineDelta(headlineFor(run("n", [["reactive", "mcc", 0.4, null]]), "reactive")),
    ).toBeNull();
    expect(baselineDelta(undefined)).toBeNull();
  });
});
