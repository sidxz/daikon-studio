import { gauss, mulberry32 } from "./prng";

export interface BoostingExample {
  xs: number[];
  ys: number[];
  /** Step boundaries from 0 to 1; every stage is constant on [bounds[j], bounds[j + 1]]. */
  bounds: number[];
  /** stages[k][j]: the fit after k rounds on interval j. stages[0] is the mean. */
  stages: number[][];
  /** pointStages[k][i]: the fit after k rounds at xs[i]. */
  pointStages: number[][];
}

const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;

/** Gradient boosting with decision stumps on 14 noisy points: six rounds, learning rate 0.55. */
export function boostingExample(): BoostingExample {
  const n = 14;
  const rounds = 6;
  const eta = 0.55;
  const rand = mulberry32(7);
  const xs = Array.from({ length: n }, (_, i) => 0.02 + (i / (n - 1)) * 0.96);
  const ys = xs.map(
    (x) => 0.5 + 0.32 * Math.sin(2 * Math.PI * 0.85 * x + 0.3) + gauss(rand) * 0.035,
  );
  const base = mean(ys);
  const stumps: { split: number; left: number; right: number }[] = [];
  let fit = ys.map(() => base);
  for (let m = 0; m < rounds; m++) {
    const residual = ys.map((y, i) => y - fit[i]);
    let best = { split: 0, left: 0, right: 0, sse: Number.POSITIVE_INFINITY };
    for (let k = 1; k < n; k++) {
      const left = residual.slice(0, k);
      const right = residual.slice(k);
      const lm = mean(left);
      const rm = mean(right);
      const sse =
        left.reduce((s, v) => s + (v - lm) ** 2, 0) + right.reduce((s, v) => s + (v - rm) ** 2, 0);
      if (sse < best.sse)
        best = { split: (xs[k - 1] + xs[k]) / 2, left: eta * lm, right: eta * rm, sse };
    }
    stumps.push(best);
    fit = fit.map((f, i) => f + (xs[i] < best.split ? best.left : best.right));
  }
  const at = (k: number, x: number) =>
    stumps.slice(0, k).reduce((v, s) => v + (x < s.split ? s.left : s.right), base);
  const bounds = [0, ...[...new Set(stumps.map((s) => s.split))].sort((a, b) => a - b), 1];
  const mids = bounds.slice(0, -1).map((b, j) => (b + bounds[j + 1]) / 2);
  const ks = Array.from({ length: rounds + 1 }, (_, k) => k);
  return {
    xs,
    ys,
    bounds,
    stages: ks.map((k) => mids.map((x) => at(k, x))),
    pointStages: ks.map((k) => xs.map((x) => at(k, x))),
  };
}
