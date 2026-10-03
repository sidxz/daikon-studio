import { mulberry32, shuffle } from "./prng";

export interface BootstrapExample {
  correct: boolean[];
  /** draws[b][k]: the k-th test example picked in redraw b (with replacement). */
  draws: number[][];
  accuracies: number[];
  interval: [number, number];
}

/** 40 test examples, 30 correct, 200 redraws, percentile 95% interval. */
export function bootstrapExample(): BootstrapExample {
  const n = 40;
  const resamples = 200;
  const correct = new Array<boolean>(n).fill(false);
  for (const i of shuffle([...Array(n).keys()], mulberry32(3)).slice(0, 30)) correct[i] = true;
  const rand = mulberry32(21);
  const draws = Array.from({ length: resamples }, () =>
    Array.from({ length: n }, () => Math.floor(rand() * n)),
  );
  const accuracies = draws.map((d) => d.filter((i) => correct[i]).length / n);
  const sorted = [...accuracies].sort((a, b) => a - b);
  const q = (p: number) =>
    sorted[Math.min(resamples - 1, Math.max(0, Math.round(p * (resamples - 1))))];
  return { correct, draws, accuracies, interval: [q(0.025), q(0.975)] };
}
