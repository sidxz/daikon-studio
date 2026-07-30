import { describe, expect, it } from "vitest";
import { type Engine, TASK_FOR_TARGET_KIND, enginesForTargetKind } from "./index";

function engine(id: string, tasks: string[]): Engine {
  return {
    id,
    version: "1.0.0",
    name: id,
    description: "",
    tasks,
    conditions: [],
    is_baseline: false,
  };
}

const ENGINES = [
  engine("regressor", ["regression"]),
  engine("classifier", ["binary_classification"]),
  engine("both", ["regression", "binary_classification"]),
];

describe("engine/target-kind translation", () => {
  it("maps every target kind the API can return", () => {
    // If the backend gains a TargetKind and this table is not updated, the
    // engine picker silently filters everything out rather than failing loudly.
    expect(Object.keys(TASK_FOR_TARGET_KIND).sort()).toEqual(["binary", "numeric"]);
  });

  it("offers regression engines for a numeric target", () => {
    expect(enginesForTargetKind(ENGINES, "numeric").map((e) => e.id)).toEqual([
      "regressor",
      "both",
    ]);
  });

  it("offers classifiers for a binary target", () => {
    expect(enginesForTargetKind(ENGINES, "binary").map((e) => e.id)).toEqual([
      "classifier",
      "both",
    ]);
  });
});
