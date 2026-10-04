import { describe, expect, it } from "vitest";
import {
  type Engine,
  PINNED_BY_PRETRAINED,
  TASK_FOR_TARGET_KIND,
  enginesForTargets,
  jointEnginesRefused,
  trainingKind,
} from "./index";

function engine(id: string, tasks: string[]): Engine {
  return {
    id,
    version: "1.0.0",
    name: id,
    description: "",
    tasks,
    conditions: [],
    is_baseline: false,
    supports_multitask: false,
    lane: "default",
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
    expect(enginesForTargets(ENGINES, [{ kind: "numeric" }]).map((e) => e.id)).toEqual([
      "regressor",
      "both",
    ]);
  });

  it("offers classifiers for a binary target", () => {
    expect(enginesForTargets(ENGINES, [{ kind: "binary" }]).map((e) => e.id)).toEqual([
      "classifier",
      "both",
    ]);
  });
});

describe("engines for several targets", () => {
  const rf = {
    ...engine("rf", ["regression", "binary_classification"]),
    supports_multitask: false,
  };
  const chemprop = {
    ...engine("chemprop", ["regression", "binary_classification"]),
    supports_multitask: true,
  };
  const regressor = engine("gp-reg", ["regression"]);

  it("offers an engine only if it supports every target's task", () => {
    expect(enginesForTargets([rf, regressor], [{ kind: "numeric" }, { kind: "binary" }])).toEqual([
      rf,
    ]);
  });

  it("withholds a joint engine from a dataset that mixes kinds, and names it", () => {
    const mixed = [{ kind: "numeric" as const }, { kind: "binary" as const }];
    expect(enginesForTargets([rf, chemprop], mixed)).toEqual([rf]);
    expect(jointEnginesRefused([rf, chemprop], mixed)).toEqual([chemprop]);
    expect(enginesForTargets([rf, chemprop], [{ kind: "binary" }, { kind: "binary" }])).toEqual([
      rf,
      chemprop,
    ]);
  });

  it("says how an engine trains several targets, and nothing for one", () => {
    expect(trainingKind(chemprop, 4)).toBe("Trains one joint model on all 4 targets.");
    expect(trainingKind(rf, 4)).toBe("Trains 4 separate models, one per target.");
    expect(trainingKind(rf, 1)).toBeNull();
  });
});

describe("PINNED_BY_PRETRAINED", () => {
  it("pins the two settings CheMeleon's checkpoint fixes", () => {
    expect(PINNED_BY_PRETRAINED.CheMeleon).toEqual({
      message_hidden_dim: 2048,
      depth: 6,
    });
  });

  it("pins values that sit inside the manifest's declared bounds", () => {
    // Why this matters: the form submits these, so an out-of-bounds pin would
    // be rejected by validate_conditions on the server.
    expect(PINNED_BY_PRETRAINED.CheMeleon.depth).toBeLessThanOrEqual(6);
    expect(PINNED_BY_PRETRAINED.CheMeleon.message_hidden_dim).toBeLessThanOrEqual(2400);
  });
});
