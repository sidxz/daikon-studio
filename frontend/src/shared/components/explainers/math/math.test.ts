import { describe, expect, it } from "vitest";
import { boostingExample } from "./boosting";
import { type BootstrapData, bootstrapLayout, niceTicks } from "./bootstrap";
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
  const data: BootstrapData = {
    metric: "RMSE",
    higherIsBetter: false,
    interval: [0.6, 0.85],
    baseline: 1.2,
    redraws: { edges: [0.5, 0.6, 0.7, 0.8, 0.9], counts: [3, 400, 550, 47] },
    compounds: Array.from({ length: 300 }, (_, i) => ({ actual: 0, predicted: i / 100 })),
    testSize: 300,
    cutoff: null,
  };

  it("shows at most 40 compounds, spread over the whole test set", () => {
    const { cells } = bootstrapLayout(data);
    expect(cells).toHaveLength(38);
    expect(Math.max(...cells.map((c) => c.size))).toBe(1);
    expect(cells[0].size).toBe(0);
  });
  it("scales dots so the tallest column fits, and never drops a nonempty bin", () => {
    const { per, dots, total } = bootstrapLayout(data);
    expect(total).toBe(1000);
    expect(per).toBe(20);
    const column = (bin: number) => dots.filter((d) => d.bin === bin);
    expect(column(0)).toHaveLength(1);
    expect(column(2)).toHaveLength(28);
    expect(
      column(2)
        .map((d) => d.level)
        .sort((a, b) => a - b),
    ).toEqual([...Array(28).keys()]);
  });
  it("puts a near baseline on the axis and a far one off it", () => {
    const near = bootstrapLayout(data);
    expect(near.baselineOff).toBe(0);
    expect(near.domain[1]).toBeGreaterThan(1.2);
    const far = bootstrapLayout({ ...data, baseline: 5 });
    expect(far.baselineOff).toBe(1);
    expect(far.domain[1]).toBeLessThan(1);
  });
  it("marks classification compounds by whether they were predicted correctly at the cutoff", () => {
    const { cells } = bootstrapLayout({
      ...data,
      cutoff: 0.3,
      compounds: [
        { actual: 1, predicted: 0.4 },
        { actual: 0, predicted: 0.4 },
      ],
    });
    expect(cells.map((c) => c.ok)).toEqual([true, false]);
  });
  it("ticks on round numbers inside the domain", () => {
    expect(niceTicks(0.48, 1.03)).toEqual([0.5, 0.6, 0.7, 0.8, 0.9, 1]);
    expect(niceTicks(0.55, 1.3)).toEqual([0.6, 0.8, 1, 1.2]);
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
