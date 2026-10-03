import { describe, expect, it } from "vitest";
import { boostingExample } from "./boosting";
import { bootstrapExample } from "./bootstrap";
import { domainExample, typicalError } from "./domain";
import { gpExample } from "./gaussian-process";
import { mulberry32, shuffle } from "./prng";
import { splitExample } from "./split";
import { seg } from "./tween";

const mse = (ys: number[], fit: number[]) =>
  ys.reduce((s, y, i) => s + (y - fit[i]) ** 2, 0) / ys.length;

describe("tween", () => {
  it("clamps window progress", () => {
    expect(seg(0.5, 0.2, 0.4)).toBe(1);
    expect(seg(0.1, 0.2, 0.4)).toBe(0);
    expect(seg(0.3, 0.2, 0.4)).toBeCloseTo(0.5);
  });
});

describe("prng", () => {
  it("is deterministic per seed and in [0, 1)", () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    for (let i = 0; i < 100; i++) {
      const v = a();
      expect(v).toBe(b());
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
  it("shuffles into a permutation", () => {
    const out = shuffle([...Array(40).keys()], mulberry32(3));
    expect([...out].sort((x, y) => x - y)).toEqual([...Array(40).keys()]);
  });
});

describe("boosting", () => {
  it("lowers the squared error every round", () => {
    const ex = boostingExample();
    expect(ex.pointStages).toHaveLength(7);
    for (let k = 1; k < ex.pointStages.length; k++) {
      expect(mse(ex.ys, ex.pointStages[k])).toBeLessThan(mse(ex.ys, ex.pointStages[k - 1]));
    }
  });
  it("spans [0, 1] with sorted step bounds", () => {
    const { bounds, stages } = boostingExample();
    expect(bounds[0]).toBe(0);
    expect(bounds.at(-1)).toBe(1);
    expect([...bounds].sort((a, b) => a - b)).toEqual(bounds);
    for (const stage of stages) expect(stage).toHaveLength(bounds.length - 1);
  });
});

describe("gaussian process", () => {
  it("starts at the prior and pins the curve near observations", () => {
    const { xs, ys, grid, states, params } = gpExample();
    for (const sd of states[0].sd) expect(sd).toBeCloseTo(params.sigma);
    const last = states[states.length - 1];
    xs.forEach((x, i) => {
      const g = Math.round(x * (grid.length - 1));
      expect(last.sd[g]).toBeLessThan(0.05);
      expect(Math.abs(last.mean[g] - ys[i])).toBeLessThan(0.03);
    });
    expect(last.sd[Math.round(0.57 * (grid.length - 1))]).toBeGreaterThan(0.15);
  });
});

describe("bootstrap", () => {
  it("reproduces the previewed interval and holds ~95% of redraws", () => {
    const { accuracies, interval } = bootstrapExample();
    expect(interval).toEqual([0.625, 0.9]);
    const inside = accuracies.filter((a) => a >= interval[0] && a <= interval[1]).length;
    expect(inside / accuracies.length).toBeGreaterThanOrEqual(0.94);
    for (const a of accuracies) expect((a * 40) % 1).toBeCloseTo(0);
  });
});

describe("split", () => {
  it("holds out 22 points either way, one whole group under scaffold", () => {
    const { points, random, scaffold } = splitExample();
    expect(random.test).toHaveLength(22);
    expect(scaffold.test).toHaveLength(22);
    expect(new Set(scaffold.test.map((i) => points[i].group)).size).toBe(1);
  });
  it("links each test point to a training point", () => {
    for (const mode of [splitExample().random, splitExample().scaffold]) {
      const test = new Set(mode.test);
      for (const link of mode.links) expect(test.has(link.train)).toBe(false);
    }
  });
  it("makes the scaffold test set less similar to training than the random one", () => {
    const { random, scaffold } = splitExample();
    expect(scaffold.medianSimilarity).toBeLessThan(random.medianSimilarity);
  });
});

describe("domain", () => {
  it("stamps two positions inside the domain and one outside", () => {
    const { stamps } = domainExample(0.3);
    expect(stamps).toHaveLength(3);
    expect(stamps[0].similarity).toBeGreaterThanOrEqual(0.3);
    expect(stamps[1].similarity).toBeGreaterThanOrEqual(0.3);
    expect(stamps[2].similarity).toBeLessThan(0.3);
  });
  it("has error falling as similarity rises", () => {
    expect(typicalError(0.2)).toBeGreaterThan(typicalError(0.8));
  });
});
