export interface GpParams {
  sigma: number;
  lengthScale: number;
  noise: number;
  priorMean: number;
}
export interface GpState {
  mean: number[];
  sd: number[];
}
export interface GpExample {
  xs: number[];
  ys: number[];
  grid: number[];
  params: GpParams;
  states: GpState[];
}

type Matrix = number[][];

function cholesky(a: Matrix): Matrix {
  const n = a.length;
  const l = a.map(() => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let q = 0; q < j; q++) s -= l[i][q] * l[j][q];
      l[i][j] = i === j ? Math.sqrt(s) : s / l[j][j];
    }
  }
  return l;
}
const forward = (l: Matrix, b: number[]) => {
  const y: number[] = [];
  b.forEach((v, i) => {
    let s = v;
    for (let q = 0; q < i; q++) s -= l[i][q] * y[q];
    y.push(s / l[i][i]);
  });
  return y;
};
const backward = (l: Matrix, y: number[]) => {
  const n = y.length;
  const x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let q = i + 1; q < n; q++) s -= l[q][i] * x[q];
    x[i] = s / l[i][i];
  }
  return x;
};

/** states[n] is the RBF posterior on `grid` after the first n observations. */
export function gpStates(xs: number[], ys: number[], grid: number[], p: GpParams): GpState[] {
  const k = (a: number, b: number) =>
    p.sigma ** 2 * Math.exp(-((a - b) ** 2) / (2 * p.lengthScale ** 2));
  return Array.from({ length: xs.length + 1 }, (_, n) => {
    if (n === 0) return { mean: grid.map(() => p.priorMean), sd: grid.map(() => p.sigma) };
    const x = xs.slice(0, n);
    const l = cholesky(x.map((a, i) => x.map((b, j) => k(a, b) + (i === j ? p.noise ** 2 : 0))));
    const alpha = backward(
      l,
      forward(
        l,
        ys.slice(0, n).map((y) => y - p.priorMean),
      ),
    );
    const mean: number[] = [];
    const sd: number[] = [];
    for (const g of grid) {
      const ks = x.map((a) => k(g, a));
      const w = forward(l, ks);
      mean.push(p.priorMean + ks.reduce((s, z, i) => s + z * alpha[i], 0));
      sd.push(Math.sqrt(Math.max(1e-9, k(g, g) - w.reduce((s, z) => s + z * z, 0))));
    }
    return { mean, sd };
  });
}

export function gpExample(): GpExample {
  const f = (x: number) => 0.5 + 0.26 * Math.sin(7 * x) + 0.08 * Math.cos(17 * x);
  const xs = [0.08, 0.26, 0.36, 0.78, 0.92];
  const ys = xs.map(f);
  const grid = Array.from({ length: 91 }, (_, i) => i / 90);
  const params = { sigma: 0.2, lengthScale: 0.085, noise: 0.012, priorMean: 0.5 };
  return { xs, ys, grid, params, states: gpStates(xs, ys, grid, params) };
}
