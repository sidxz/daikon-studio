import { mulberry32, shuffle } from "./prng";

/** A Scorecard's real interval and the redraws it was read from, as the figure needs them. */
export interface BootstrapData {
  /** As the card shows it, e.g. "RMSE". */
  metric: string;
  higherIsBetter: boolean;
  interval: [number, number];
  baseline: number;
  /** The primary metric on each redraw, binned; `edges` is one longer than `counts`. */
  redraws: { edges: number[]; counts: number[] };
  /** Test compounds as the card carries them (`parity`, subsampled past 4,000). */
  compounds: { actual: number; predicted: number }[];
  /** Every test compound, not just the ones carried. */
  testSize: number;
  /** The decision cutoff for classification; null for regression. */
  cutoff: number | null;
}

export interface BootstrapLayout {
  /** At most 40 test compounds: `ok` for classification, `size` (0..1, by error) for regression. */
  cells: { ok: boolean; size: number }[];
  /** Redraws each dot stands for, so the tallest column fits the figure. */
  per: number;
  total: number;
  /** One per dot, in the order they land: its bin and its height in that bin's column. */
  dots: { bin: number; level: number }[];
  domain: [number, number];
  ticks: number[];
  /** -1 or 1 when the baseline is too far from the redraws to share an axis with them. */
  baselineOff: -1 | 0 | 1;
}

const MAX_CELLS = 40;
const MAX_STACK = 30;
/** Below this share of the axis the redraws crush into a sliver, so the baseline goes off scale. */
const MIN_SHARE = 0.3;

export function bootstrapLayout(d: BootstrapData): BootstrapLayout {
  const stride = Math.max(1, Math.ceil(d.compounds.length / MAX_CELLS));
  const shown = d.compounds.filter((_, i) => i % stride === 0).slice(0, MAX_CELLS);
  const errors = shown.map((c) => Math.abs(c.predicted - c.actual));
  const worst = Math.max(...errors, 0) || 1;
  const cells = shown.map((c, i) => ({
    ok: d.cutoff != null && c.predicted >= d.cutoff === c.actual >= 0.5,
    size: Math.sqrt(errors[i] / worst),
  }));

  const { edges, counts } = d.redraws;
  const total = counts.reduce((a, b) => a + b, 0);
  const tallest = Math.max(...counts, 1);
  const per =
    [1, 2, 5, 10, 20, 50, 100].find((p) => tallest / p <= MAX_STACK) ??
    Math.ceil(tallest / MAX_STACK);
  // ponytail: a bin of 1 to `per` redraws still draws one dot, so the thinnest
  // tails read slightly heavy; the interval band is the exact one either way.
  const bins = counts.flatMap((n, bin) =>
    Array<number>(n > 0 ? Math.max(1, Math.round(n / per)) : 0).fill(bin),
  );
  const height = new Array<number>(counts.length).fill(0);
  const dots = shuffle(bins, mulberry32(7)).map((bin) => ({ bin, level: height[bin]++ }));

  const lo = edges[0];
  const hi = edges[edges.length - 1];
  const wideLo = Math.min(lo, d.baseline);
  const wideHi = Math.max(hi, d.baseline);
  const fits = (hi - lo) / (wideHi - wideLo) >= MIN_SHARE;
  const [a, b] = fits ? [wideLo, wideHi] : [lo, hi];
  const pad = (b - a) * 0.04;
  const domain: [number, number] = [a - pad, b + pad];
  const baselineOff = fits ? 0 : d.baseline < lo ? -1 : 1;

  return { cells, per, total, dots, domain, ticks: niceTicks(...domain), baselineOff };
}

/** About five round-numbered ticks inside [lo, hi]. */
export function niceTicks(lo: number, hi: number): number[] {
  const raw = (hi - lo) / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const e = raw / mag;
  // d3's thresholds: the step nearest five ticks, not the first one at or above it.
  const step = (e >= Math.sqrt(50) ? 10 : e >= Math.sqrt(10) ? 5 : e >= Math.SQRT2 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step)
    out.push(Number(v.toFixed(10)));
  return out;
}
