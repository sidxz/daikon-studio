# Explainer Figures and Navigation Icons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight animated, collapsible explainer figures (A1–A5, B, C, D) at their approved placements, and replace the sidebar's lucide icons with seven colored custom glyphs.

**Architecture:** Pure math modules compute each figure's data from fixed seeds (tested in isolation). A `useTimeline` hook turns view/hover/replay into a number `t` in [0, 1]; each figure is a pure `({ t }) => <svg>` component. An `Explainer` disclosure (Radix Collapsible + a persisted zustand store) wraps every figure. The icons are plain SVG components with `currentColor`, colored through new `text-icon-*` utilities.

**Tech Stack:** Next 16, React 19, TypeScript strict, Tailwind v4 (tokens from `@structflo/daikon-design-tokens`), Radix (`radix-ui`), zustand `persist`, vitest + Testing Library (jsdom), biome.

**Spec:** `docs/superpowers/specs/2026-10-03-explainer-figures-design.md`. The visual reference to port from is `docs/superpowers/specs/2026-10-03-explainer-figures-preview.html` (cited below as "preview", with line numbers).

## Global Constraints

- No new dependency. SVG + React + CSS only; no WebGL, no animation library.
- Every figure renders its final frame (`t = 1`) on first render, on the server, and under `prefers-reduced-motion: reduce`.
- It plays once when 35% of the figure is in view, replays on `pointerenter` (only when idle) and on the Replay button.
- Each disclosure is open when nothing is stored for it; it is remembered per browser and per figure id under the localStorage key `ds-explainers`.
- Colors only through tokens: measured `chart-1`, test/new `score-fair`, messages `chart-2`, model output `foreground` drawn as an open ring, text `secondary-foreground`, structure `muted-foreground/50`.
- Copy: American spelling, sentence case, no em-dash clause chains (`docs/copy-audit.md`). Captions are the preview's captions, including the sentence that says where each figure simplifies.
- Leave the logo (`logo-mark.tsx`) unchanged.
- Domain threshold comes from `IN_DOMAIN_FLOOR` (0.3), passed in as a prop. `shared/` never imports from `features/`.
- Icon hues: Datasets `chart-1`, Protocols `ds-success`, Sweeps `chart-3`, Runs `#b45309` / `#fbbf24`, Collections `#db2777` / `#f472b6`, Engines `ds-score-good`, Runners `#0891b2` / `#22d3ee`, Dashboard `ds-text-secondary`.
- Frontend commands run from `frontend/`: `pnpm test`, `pnpm lint`, `pnpm exec tsc --noEmit`, `pnpm build`.

## Review Focus

1. **Collapsed, then opened:** a figure whose disclosure was closed must still play when opened. Radix unmounts closed content, so the timeline lives inside the content and starts on mount. Test it in Task 2.
2. **Engine with no figure:** a new engine id, or an engine list still loading, must render no explainer and must not throw. Test it in Task 4.
3. **Reduced motion while in view:** never schedules a frame and stays at `t = 1`. Test it in Task 2.
4. **Unusual values reaching the drawing:** `t` slightly outside [0, 1] (rAF timing) or `t = 0` must not produce `NaN` in any SVG attribute. Test it in Task 3 (every figure at `t` in {0, 0.37, 1, 1.0001}).
5. **Theme switch:** colors must follow `data-theme` with no re-render, so classes only, never hex literals in figure components. Enforced by a grep step in Task 7.

---

## File structure

```
frontend/src/shared/components/explainers/
  math/tween.ts            clamp, seg, ease, easeInOut, lerp
  math/prng.ts             mulberry32, gauss, shuffle
  math/boosting.ts         boostingExample()
  math/gaussian-process.ts gpExample(), gpStates()
  math/bootstrap.ts        bootstrapExample()
  math/split.ts            splitExample(), nearestTraining(), median()
  math/domain.ts           domainExample(), typicalError(), nearestPoint()
  math/math.test.ts        all math tests
  use-timeline.ts          useTimeline()
  explainer-store.ts       useExplainerStore
  explainer.tsx            <Explainer>
  explainer.test.tsx       hook + disclosure tests
  figures/styles.ts        shared class strings
  figures/forest.tsx  boosting.tsx  gaussian-process.tsx  message-passing.tsx  attention.tsx
  figures/split.tsx  bootstrap.tsx  domain.tsx
  figures/figures.test.tsx smoke tests
frontend/src/shared/components/icons/nav-icons.tsx
frontend/src/shared/components/icons/nav-icons.test.tsx
frontend/src/features/engines/components/engine-explainer.tsx (+ .test.tsx)
Modified: features/engines/types/index.ts, engine-catalogue.tsx, train-protocol-form.tsx,
  dataset-wizard.tsx, dataset-profile-view.tsx, scorecard-view.tsx, scorecard-diagnostics.tsx,
  run-detail.tsx, shared/lib/navigation.ts, nav-main.tsx, command-palette.tsx,
  app/(dashboard)/page.tsx, app/globals.css
```

---

### Task 1: Math modules

**Files:**
- Create: `frontend/src/shared/components/explainers/math/{tween,prng,boosting,gaussian-process,bootstrap,split,domain}.ts`
- Test: `frontend/src/shared/components/explainers/math/math.test.ts`

**Interfaces:**
- Produces: `clamp(x, lo=0, hi=1)`, `seg(t, a, b)`, `ease(x)`, `easeInOut(x)`, `lerp(a, b, p)`; `mulberry32(seed): () => number`, `gauss(rand)`, `shuffle<T>(items, rand): T[]`; `boostingExample(): BoostingExample`; `gpExample(): GpExample`; `bootstrapExample(): BootstrapExample`; `splitExample(): SplitExample`; `median(values)`; `domainExample(threshold): DomainExample`; `typicalError(s)`; `nearestPoint(points, x, y)`. All types are exported from their modules as shown below.

- [ ] **Step 1: Write the failing tests** (`math/math.test.ts`)

```ts
import { describe, expect, it } from "vitest";
import { bootstrapExample } from "./bootstrap";
import { boostingExample } from "./boosting";
import { domainExample, typicalError } from "./domain";
import { gpExample } from "./gaussian-process";
import { mulberry32, shuffle } from "./prng";
import { splitExample } from "./split";
import { seg } from "./tween";

const mse = (ys: number[], fit: number[]) => ys.reduce((s, y, i) => s + (y - fit[i]) ** 2, 0) / ys.length;

describe("tween", () => {
  it("clamps window progress", () => {
    expect(seg(0.5, 0.2, 0.4)).toBe(1);
    expect(seg(0.1, 0.2, 0.4)).toBe(0);
    expect(seg(0.3, 0.2, 0.4)).toBeCloseTo(0.5);
  });
});

describe("prng", () => {
  it("is deterministic per seed and in [0, 1)", () => {
    const a = mulberry32(7), b = mulberry32(7);
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
    const last = states.at(-1)!;
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd frontend && pnpm test src/shared/components/explainers/math`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement the modules**

`math/tween.ts`
```ts
/** Timeline helpers. Every explainer figure is a pure function of t in [0, 1]. */
export const clamp = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
/** Progress of t through the window [a, b], clamped to [0, 1]. */
export const seg = (t: number, a: number, b: number) => clamp((t - a) / (b - a));
export const ease = (x: number) => 1 - (1 - x) ** 3;
export const easeInOut = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
```

`math/prng.ts`
```ts
/** Seeded PRNG (mulberry32), so a figure draws the same picture on every load. */
export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draw (Box–Muller). */
export function gauss(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export function shuffle<T>(items: readonly T[], rand: () => number): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
```

`math/boosting.ts`
```ts
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
  const n = 14, rounds = 6, eta = 0.55;
  const rand = mulberry32(7);
  const xs = Array.from({ length: n }, (_, i) => 0.02 + (i / (n - 1)) * 0.96);
  const ys = xs.map((x) => 0.5 + 0.32 * Math.sin(2 * Math.PI * 0.85 * x + 0.3) + gauss(rand) * 0.035);
  const base = mean(ys);
  const stumps: { split: number; left: number; right: number }[] = [];
  let fit = ys.map(() => base);
  for (let m = 0; m < rounds; m++) {
    const residual = ys.map((y, i) => y - fit[i]);
    let best = { split: 0, left: 0, right: 0, sse: Number.POSITIVE_INFINITY };
    for (let k = 1; k < n; k++) {
      const left = residual.slice(0, k), right = residual.slice(k);
      const lm = mean(left), rm = mean(right);
      const sse = left.reduce((s, v) => s + (v - lm) ** 2, 0) + right.reduce((s, v) => s + (v - rm) ** 2, 0);
      if (sse < best.sse) best = { split: (xs[k - 1] + xs[k]) / 2, left: eta * lm, right: eta * rm, sse };
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
```

`math/gaussian-process.ts`
```ts
export interface GpParams { sigma: number; lengthScale: number; noise: number; priorMean: number }
export interface GpState { mean: number[]; sd: number[] }
export interface GpExample { xs: number[]; ys: number[]; grid: number[]; params: GpParams; states: GpState[] }

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
  const k = (a: number, b: number) => p.sigma ** 2 * Math.exp(-((a - b) ** 2) / (2 * p.lengthScale ** 2));
  return Array.from({ length: xs.length + 1 }, (_, n) => {
    if (n === 0) return { mean: grid.map(() => p.priorMean), sd: grid.map(() => p.sigma) };
    const x = xs.slice(0, n);
    const l = cholesky(x.map((a, i) => x.map((b, j) => k(a, b) + (i === j ? p.noise ** 2 : 0))));
    const alpha = backward(l, forward(l, ys.slice(0, n).map((y) => y - p.priorMean)));
    const mean: number[] = [], sd: number[] = [];
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
```

`math/bootstrap.ts`
```ts
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
  const n = 40, resamples = 200;
  const correct = new Array<boolean>(n).fill(false);
  for (const i of shuffle([...Array(n).keys()], mulberry32(3)).slice(0, 30)) correct[i] = true;
  const rand = mulberry32(21);
  const draws = Array.from({ length: resamples }, () => Array.from({ length: n }, () => Math.floor(rand() * n)));
  const accuracies = draws.map((d) => d.filter((i) => correct[i]).length / n);
  const sorted = [...accuracies].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.min(resamples - 1, Math.max(0, Math.round(p * (resamples - 1))))];
  return { correct, draws, accuracies, interval: [q(0.025), q(0.975)] };
}
```

`math/split.ts`
```ts
import { gauss, mulberry32, shuffle } from "./prng";

export interface SplitPoint { x: number; y: number; group: number }
export interface SplitLink { test: number; train: number; distance: number }
export interface SplitMode { test: number[]; links: SplitLink[]; medianSimilarity: number }
export interface SplitExample { points: SplitPoint[]; random: SplitMode; scaffold: SplitMode }

/** On-screen distance to a 0..1 stand-in for Tanimoto similarity. Illustrative only. */
export const similarityAt = (distance: number) => Math.exp(-distance / 45);

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
}

function mode(points: SplitPoint[], test: number[]): SplitMode {
  const held = new Set(test);
  const links = test.map((i) => {
    let train = -1, distance = Number.POSITIVE_INFINITY;
    points.forEach((p, j) => {
      if (held.has(j)) return;
      const d = Math.hypot(points[i].x - p.x, points[i].y - p.y);
      if (d < distance) { distance = d; train = j; }
    });
    return { test: i, train, distance };
  });
  return { test, links, medianSimilarity: median(links.map((l) => similarityAt(l.distance))) };
}

/** 110 points in five groups; one fifth held out at random, or one whole group (scaffold). */
export function splitExample(): SplitExample {
  const rand = mulberry32(11);
  const groups: [number, number, number][] = [[78, 70, 24], [196, 52, 20], [318, 112, 22], [112, 196, 18], [262, 206, 26]];
  const points: SplitPoint[] = [];
  groups.forEach(([cx, cy, n], group) => {
    for (let i = 0; i < n; i++) points.push({ x: cx + gauss(rand) * 17, y: cy + gauss(rand) * 14, group });
  });
  const random = shuffle([...points.keys()], mulberry32(5)).slice(0, 22);
  const scaffold = [...points.keys()].filter((i) => points[i].group === 2);
  return { points, random: mode(points, random), scaffold: mode(points, scaffold) };
}
```

`math/domain.ts`
```ts
import { gauss, mulberry32 } from "./prng";
import { lerp } from "./tween";

export interface PathPoint { s: number; x: number; y: number; similarity: number }
export interface DomainExample {
  training: [number, number][];
  /** Distance at which similarity falls to the threshold: the radius of the domain region. */
  reach: number;
  start: [number, number];
  end: [number, number];
  path: PathPoint[];
  stamps: PathPoint[];
}

const SCALE = 45;
/** Illustrative typical error at a given similarity (0..1). Not fitted to any protocol. */
export const typicalError = (s: number) => 0.12 + 0.86 * (1 - s) ** 2.2;

export function nearestPoint(points: [number, number][], x: number, y: number) {
  let distance = Number.POSITIVE_INFINITY, index = 0;
  points.forEach(([a, b], j) => {
    const d = Math.hypot(x - a, y - b);
    if (d < distance) { distance = d; index = j; }
  });
  return { distance, index };
}

/** A query walks away from three training clusters; stamps at start, near the edge, and outside. */
export function domainExample(threshold: number): DomainExample {
  const rand = mulberry32(9);
  const training: [number, number][] = [];
  for (const [cx, cy, n, s] of [[86, 92, 24, 16], [146, 172, 26, 18], [206, 96, 14, 13]]) {
    for (let i = 0; i < n; i++) training.push([cx + gauss(rand) * s, cy + gauss(rand) * s * 0.85]);
  }
  const start: [number, number] = [118, 132], end: [number, number] = [318, 214];
  const path = Array.from({ length: 241 }, (_, i) => {
    const s = i / 240, x = lerp(start[0], end[0], s), y = lerp(start[1], end[1], s);
    return { s, x, y, similarity: Math.exp(-nearestPoint(training, x, y).distance / SCALE) };
  });
  const edge = path.findIndex((p) => p.similarity < threshold + 0.06);
  const stamps = [0, Math.max(1, edge), path.length - 1].map((i) => path[i]);
  return { training, reach: -SCALE * Math.log(threshold), start, end, path, stamps };
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd frontend && pnpm test src/shared/components/explainers/math`
Expected: PASS (all describe blocks). If the bootstrap interval differs from `[0.625, 0.9]`, the port changed the draw order. Fix the port; do not edit the test.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/shared/components/explainers/math
git commit -m "feat(explainers): seeded math for the explainer figures"
```

---

### Task 2: Timeline hook, disclosure store, Explainer

**Files:**
- Create: `explainers/use-timeline.ts`, `explainers/explainer-store.ts`, `explainers/explainer.tsx`
- Test: `explainers/explainer.test.tsx`

**Interfaces:**
- Produces: `useTimeline<T extends Element>(durationMs: number): { ref: RefObject<T | null>; t: number; replay: () => void }`; `useExplainerStore` with `closed: Record<string, boolean>` and `setOpen(id: string, open: boolean)`; `<Explainer id label? caption durationMs replayKey? children={(t) => ReactNode} />`.

- [ ] **Step 1: Write the failing tests** (`explainer.test.tsx`)

```tsx
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Explainer } from "./explainer";
import { useExplainerStore } from "./explainer-store";
import { useTimeline } from "./use-timeline";

let frames: FrameRequestCallback[] = [];
let observe: IntersectionObserverCallback | null = null;
const disconnect = vi.fn();
const flush = (now: number) =>
  act(() => {
    const run = frames;
    frames = [];
    for (const cb of run) cb(now);
  });
const enterView = () => act(() => observe?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver));
const reduceMotion = (on: boolean) =>
  vi.spyOn(window, "matchMedia").mockImplementation((q: string) => ({ matches: on && q.includes("reduce"), media: q, onchange: null, addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {}, dispatchEvent: () => false }));

beforeEach(() => {
  frames = [];
  observe = null;
  disconnect.mockClear();
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal("IntersectionObserver", class {
    constructor(cb: IntersectionObserverCallback) { observe = cb; }
    observe() {}
    unobserve() {}
    disconnect() { disconnect(); }
  });
  reduceMotion(false);
  useExplainerStore.setState({ closed: {} });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function Probe() {
  const { ref, t } = useTimeline<HTMLDivElement>(1000);
  return <div ref={ref} data-testid="probe" data-t={t} />;
}
const tOf = () => Number(screen.getByTestId("probe").dataset.t);

describe("useTimeline", () => {
  it("rests on the final frame until the figure is in view", () => {
    render(<Probe />);
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
  });

  it("plays once from 0 to 1 when in view, then stops observing", () => {
    render(<Probe />);
    enterView();
    expect(tOf()).toBe(0);
    flush(100);
    flush(600);
    expect(tOf()).toBeCloseTo(0.5);
    flush(1200);
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
    expect(disconnect).toHaveBeenCalled();
  });

  it("never animates under reduced motion", () => {
    reduceMotion(true);
    render(<Probe />);
    enterView();
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
  });

  it("replays on pointer enter when idle", () => {
    render(<Probe />);
    fireEvent.pointerEnter(screen.getByTestId("probe"));
    expect(tOf()).toBe(0);
  });
});

describe("Explainer", () => {
  const renderIt = () =>
    render(
      <Explainer id="forest" caption="Trees vote." durationMs={1000}>
        {(t) => <svg role="img" aria-label="figure" data-t={t} />}
      </Explainer>,
    );

  it("is open on first sight", () => {
    renderIt();
    expect(screen.getByRole("img", { name: "figure" })).toBeInTheDocument();
    expect(screen.getByText("Trees vote.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /replay/i })).toBeInTheDocument();
  });

  it("remembers being closed, per figure id", () => {
    const { unmount } = renderIt();
    fireEvent.click(screen.getByRole("button", { name: /how this works/i }));
    expect(useExplainerStore.getState().closed.forest).toBe(true);
    unmount();
    renderIt();
    expect(screen.queryByRole("img", { name: "figure" })).not.toBeInTheDocument();
  });

  it("plays when reopened after being closed", () => {
    useExplainerStore.setState({ closed: { forest: true } });
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /how this works/i }));
    enterView();
    expect(Number(screen.getByRole("img", { name: "figure" }).dataset.t)).toBe(0);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd frontend && pnpm test src/shared/components/explainers/explainer.test.tsx`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement**

`use-timeline.ts`
```ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * The clock behind every explainer figure. `t` runs 0 -> 1 once, the first
 * time the figure is a third in view, and again on pointer-enter (when idle)
 * or replay(). It rests at 1, so the first render, a server render and
 * reduced motion all show the complete final frame.
 */
export function useTimeline<T extends Element>(durationMs: number) {
  const ref = useRef<T>(null);
  const [t, setT] = useState(1);
  const frame = useRef(0);

  const replay = useCallback(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    cancelAnimationFrame(frame.current);
    let start: number | null = null;
    const step = (now: number) => {
      start ??= now;
      const next = Math.min(1, (now - start) / durationMs);
      setT(next);
      frame.current = next < 1 ? requestAnimationFrame(step) : 0;
    };
    setT(0);
    frame.current = requestAnimationFrame(step);
  }, [durationMs]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onEnter = () => {
      if (frame.current === 0) replay();
    };
    el.addEventListener("pointerenter", onEnter);
    let observer: IntersectionObserver | undefined;
    if (typeof IntersectionObserver !== "undefined") {
      observer = new IntersectionObserver(
        (entries) => {
          if (!entries.some((e) => e.isIntersecting)) return;
          observer?.disconnect();
          replay();
        },
        { threshold: 0.35 },
      );
      observer.observe(el);
    }
    return () => {
      el.removeEventListener("pointerenter", onEnter);
      observer?.disconnect();
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [replay]);

  return { ref, t, replay };
}
```

`explainer-store.ts`
```ts
"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ExplainerState {
  /** Figure ids this browser has closed. Absent means open, so every figure is open on first sight. */
  closed: Record<string, boolean>;
  setOpen: (id: string, open: boolean) => void;
}

// ponytail: persisted state can differ from the server render, but every
// explainer sits under data that loads client-side, so none renders on the
// server today. If one ever does, add skipHydration and rehydrate in an effect.
export const useExplainerStore = create<ExplainerState>()(
  persist(
    (set) => ({
      closed: {},
      setOpen: (id, open) =>
        set((state) => ({
          closed: open
            ? Object.fromEntries(Object.entries(state.closed).filter(([key]) => key !== id))
            : { ...state.closed, [id]: true },
        })),
    }),
    { name: "ds-explainers" },
  ),
);
```

`explainer.tsx`
```tsx
"use client";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/shared/components/ui/collapsible";
import { ChevronRight, RotateCcw } from "lucide-react";
import { type ReactNode, useEffect, useRef } from "react";
import { useExplainerStore } from "./explainer-store";
import { useTimeline } from "./use-timeline";

interface ExplainerProps {
  /** Persistence key, one per figure: closing it once closes it everywhere it appears. */
  id: string;
  label?: string;
  caption: ReactNode;
  durationMs: number;
  /** Replays the figure when this changes, e.g. the selected split strategy. */
  replayKey?: string;
  children: (t: number) => ReactNode;
}

export function Explainer({ id, label = "How this works", caption, durationMs, replayKey, children }: ExplainerProps) {
  const open = useExplainerStore((state) => !state.closed[id]);
  const setOpen = useExplainerStore((state) => state.setOpen);
  return (
    <Collapsible open={open} onOpenChange={(next) => setOpen(id, next)} className="space-y-3">
      <CollapsibleTrigger className="group inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline">
        <ChevronRight className="size-4 transition-transform group-data-[state=open]:rotate-90 motion-reduce:transition-none" />
        {label}
      </CollapsibleTrigger>
      {/* Radix unmounts closed content, so the timeline starts on open. */}
      <CollapsibleContent>
        <ExplainerFigure caption={caption} durationMs={durationMs} replayKey={replayKey}>
          {children}
        </ExplainerFigure>
      </CollapsibleContent>
    </Collapsible>
  );
}

function ExplainerFigure({ caption, durationMs, replayKey, children }: Omit<ExplainerProps, "id" | "label">) {
  const { ref, t, replay } = useTimeline<HTMLDivElement>(durationMs);
  const shownKey = useRef(replayKey);
  useEffect(() => {
    if (shownKey.current === replayKey) return;
    shownKey.current = replayKey;
    replay();
  }, [replayKey, replay]);

  return (
    <div className="space-y-2 border-t pt-3">
      <div ref={ref}>{children(t)}</div>
      <div className="flex items-start justify-between gap-3">
        <p className="max-w-prose text-xs text-muted-foreground">{caption}</p>
        <button
          type="button"
          onClick={replay}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <RotateCcw className="size-3" aria-hidden="true" />
          Replay
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd frontend && pnpm test src/shared/components/explainers/explainer.test.tsx`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/shared/components/explainers/{use-timeline.ts,explainer-store.ts,explainer.tsx,explainer.test.tsx}
git commit -m "feat(explainers): timeline hook and remembered 'How this works' disclosure"
```

---

### Task 3: The eight figures

**Files:**
- Create: `explainers/figures/styles.ts` and the eight figure files listed above
- Test: `explainers/figures/figures.test.tsx`

**Interfaces:**
- Consumes: Task 1 math and tween.
- Produces (each file exports a duration, a component, and a caption):
  - `forest.tsx`: `FOREST_MS = 5600`, `ForestFigure({ t })`, `FOREST_CAPTION`
  - `boosting.tsx`: `BOOSTING_MS = 7800`, `BoostingFigure({ t })`, `BOOSTING_CAPTION`
  - `gaussian-process.tsx`: `GP_MS = 7000`, `GaussianProcessFigure({ t })`, `GP_CAPTION`
  - `message-passing.tsx`: `MESSAGE_PASSING_MS = 7600`, `MessagePassingFigure({ t })`, `MESSAGE_PASSING_CAPTION`
  - `attention.tsx`: `ATTENTION_MS = 7200`, `AttentionFigure({ t })`, `ATTENTION_CAPTION`
  - `split.tsx`: `SPLIT_MS = 5400`, `SplitFigure({ t, strategy: "scaffold" | "random" })`, `splitCaption(strategy)`
  - `bootstrap.tsx`: `BOOTSTRAP_MS = 9800`, `BootstrapFigure({ t, baseline })`, `BootstrapExplainer({ startOutside: boolean })` (wraps `Explainer` and owns the two-option baseline toggle: 0.70 inside, 0.55 outside)
  - `domain.tsx`: `DOMAIN_MS = 7800`, `DomainFigure({ t, threshold })`, `domainCaption(where: "diagnostics" | "triage")`

**Porting rules** (preview function → component; geometry, timing windows and labels unchanged):

| Component | Port from preview |
|---|---|
| ForestFigure | `figForest`, lines 633–695 |
| BoostingFigure | `figBoost` drawing, lines 698–760 (data now from `boostingExample()`) |
| GaussianProcessFigure | `figGP` drawing, lines 763–827 (data from `gpExample()`) |
| MessagePassingFigure | `figMPNN`, lines 830–878 |
| AttentionFigure | `figAttention`, lines 881–931 |
| SplitFigure | `figSplit` drawing, lines 934–1023 (data from `splitExample()`) |
| BootstrapFigure | `figBootstrap`, lines 1026–1106 (data from `bootstrapExample()`) |
| DomainFigure | `figDomain`, lines 1109–1177 (data from `domainExample(threshold)`) |

1. Every preview `mk(...)` becomes JSX. Every per-frame `setAttribute` becomes a value computed from `t` during render. Wrap data in `useMemo(() => example(), [])` (or `[threshold]`).
2. Preview CSS classes map to `styles.ts`: `.t` → `S.text`, `.t-s` → `S.small`, `.t-fg` → `S.textStrong`, `.ax`/`.edge` → `S.line`, `.node` → `S.node`, `.node-on` → `S.nodeOn`, `.m-fill` → `S.measured`, `.m-stroke` → `S.measuredLine`, `.h-fill` → `S.held`, `.h-stroke` → `S.heldLine`, `.h-ring` → `S.heldRing`, `.p-fill` → `S.pulse`, `.fit` → `S.fit`, `.pred` → `S.pred`, `.band` → `S.band`, `.box` → `S.box`, `.box-on` → `S.boxOn`, `.fg-stroke` → `S.ink`, `.fg-fill` → `S.inkFill`, `.card-fill` → `S.cardFill`.
3. Stroke widths, dash arrays and radii stay as SVG attributes (`strokeWidth`, `strokeDasharray`, `r`).
4. Elements the preview hides with `opacity 0` render with `opacity={0}`; they are not omitted, so the element count stays stable while animating.
5. Each `<svg>` gets `viewBox` from the preview, `className="block h-auto w-full overflow-visible"`, `role="img"`, and the preview's `aria-label` verbatim. The svg is the only element with a role.
6. HTML text from the preview moves into the component: Split's caption is returned by `splitCaption`, and the bootstrap verdict line becomes JSX below the svg.
7. Captions are the preview's `.fig-cap` text verbatim, except `domainCaption("triage")`, which ends "The scorecard's diagnostics show your protocol's measured version." instead of "the chart below this panel...".

`figures/styles.ts`
```ts
/** Class strings shared by every figure, all from design tokens, so theme switches need no JS. */
export const S = {
  text: "fill-secondary-foreground font-mono text-[10px]",
  textStrong: "fill-foreground font-mono text-[10px]",
  small: "fill-secondary-foreground font-mono text-[9px]",
  line: "fill-none stroke-muted-foreground/50",
  node: "fill-card stroke-muted-foreground/50",
  nodeOn: "fill-card stroke-chart-1",
  measured: "fill-chart-1",
  measuredLine: "fill-none stroke-chart-1",
  held: "fill-score-fair",
  heldLine: "fill-none stroke-score-fair",
  heldRing: "fill-card stroke-score-fair",
  pulse: "fill-chart-2",
  fit: "fill-none stroke-foreground",
  pred: "fill-card stroke-foreground",
  band: "fill-chart-1/15",
  box: "fill-card stroke-muted-foreground/50",
  boxOn: "fill-accent stroke-chart-1",
  ink: "fill-none stroke-foreground",
  inkFill: "fill-foreground",
  cardFill: "fill-card",
} as const;
```

- [ ] **Step 1: Write the failing smoke test** (`figures/figures.test.tsx`)

```tsx
import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { AttentionFigure } from "./attention";
import { BoostingFigure } from "./boosting";
import { BootstrapFigure } from "./bootstrap";
import { DomainFigure } from "./domain";
import { ForestFigure } from "./forest";
import { GaussianProcessFigure } from "./gaussian-process";
import { MessagePassingFigure } from "./message-passing";
import { SplitFigure } from "./split";

const FIGURES: [string, (t: number) => ReactElement][] = [
  ["forest", (t) => <ForestFigure t={t} />],
  ["boosting", (t) => <BoostingFigure t={t} />],
  ["gaussian process", (t) => <GaussianProcessFigure t={t} />],
  ["message passing", (t) => <MessagePassingFigure t={t} />],
  ["attention", (t) => <AttentionFigure t={t} />],
  ["split, scaffold", (t) => <SplitFigure t={t} strategy="scaffold" />],
  ["split, random", (t) => <SplitFigure t={t} strategy="random" />],
  ["bootstrap, inside", (t) => <BootstrapFigure t={t} baseline={0.7} />],
  ["bootstrap, outside", (t) => <BootstrapFigure t={t} baseline={0.55} />],
  ["domain", (t) => <DomainFigure t={t} threshold={0.3} />],
];

describe.each(FIGURES)("%s figure", (_, draw) => {
  it.each([0, 0.37, 1, 1.0001])("draws a labeled image with no NaN at t=%s", (t) => {
    const { container } = render(draw(t));
    expect(screen.getByRole("img").getAttribute("aria-label")).toBeTruthy();
    expect(container.innerHTML).not.toMatch(/NaN|Infinity|undefined/);
  });
});

describe("BootstrapFigure", () => {
  it("says within noise when the baseline is inside the interval", () => {
    render(<BootstrapFigure t={1} baseline={0.7} />);
    expect(screen.getByText("Within noise")).toBeInTheDocument();
  });
  it("says beats the baseline when it is outside", () => {
    render(<BootstrapFigure t={1} baseline={0.55} />);
    expect(screen.getByText("Beats the baseline")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify failure.** Run `cd frontend && pnpm test src/shared/components/explainers/figures`. Expected: FAIL, modules not found.

- [ ] **Step 3: Write `styles.ts`, then port the eight figures** following the rules and table above, one file at a time. After each file, run the smoke test filtered to it (`pnpm test figures -t forest`).

- [ ] **Step 4: Run all figure tests.** Run `cd frontend && pnpm test src/shared/components/explainers`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/shared/components/explainers/figures
git commit -m "feat(explainers): eight figures ported from the approved preview"
```

---

### Task 4: Engine explainers (A1–A5)

**Files:**
- Modify: `frontend/src/features/engines/types/index.ts` (append)
- Create: `frontend/src/features/engines/components/engine-explainer.tsx`, `engine-explainer.test.tsx`
- Modify: `frontend/src/features/engines/components/engine-catalogue.tsx` (CardContent, before "Settings")
- Modify: `frontend/src/features/protocols/components/train-protocol-form.tsx` (after the Engine `</Select>`)

**Interfaces:**
- Consumes: `Explainer`, and the A-figure exports from Task 3.
- Produces: `ENGINE_FIGURE: Record<string, EngineFigure>`, `<EngineExplainer engineId />`.

- [ ] **Step 1: Failing test** (`engine-explainer.test.tsx`)

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EngineExplainer } from "./engine-explainer";

describe("EngineExplainer", () => {
  it.each([
    ["ecfp4-randomforest", /decision trees/i],
    ["ecfp4-lightgbm", /gradient boosting/i],
    ["tanimoto-gp", /gaussian process/i],
    ["chemprop-dmpnn", /seven-node graph/i],
    ["molformer-xl", /seven tokens/i],
  ])("shows the right figure for %s", (id, label) => {
    render(<EngineExplainer engineId={id} />);
    expect(screen.getByRole("img", { name: label })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /how it learns/i })).toBeInTheDocument();
  });

  it("renders nothing for an engine with no figure", () => {
    const { container } = render(<EngineExplainer engineId="some-future-engine" />);
    expect(container).toBeEmptyDOMElement();
  });
});
```

- [ ] **Step 2: Run to verify failure.** `pnpm test engine-explainer` fails.

- [ ] **Step 3: Implement**

Append to `features/engines/types/index.ts`:
```ts
/**
 * Which "How it learns" figure explains each engine. Hardcoded engine
 * knowledge again, kept beside the other two maps so an engine change shows
 * it to the reviewer. An id missing here shows no figure, never a wrong one.
 */
export type EngineFigure = "forest" | "boosting" | "gaussian-process" | "message-passing" | "attention";

export const ENGINE_FIGURE: Record<string, EngineFigure> = {
  "ecfp4-randomforest": "forest",
  "ecfp4-xgboost": "boosting",
  "ecfp4-lightgbm": "boosting",
  "descriptors-xgboost": "boosting",
  "tanimoto-gp": "gaussian-process",
  "chemprop-dmpnn": "message-passing",
  "molformer-xl": "attention",
};
```

`engine-explainer.tsx`
```tsx
"use client";

import { Explainer } from "@/shared/components/explainers/explainer";
import { ATTENTION_CAPTION, ATTENTION_MS, AttentionFigure } from "@/shared/components/explainers/figures/attention";
import { BOOSTING_CAPTION, BOOSTING_MS, BoostingFigure } from "@/shared/components/explainers/figures/boosting";
import { FOREST_CAPTION, FOREST_MS, ForestFigure } from "@/shared/components/explainers/figures/forest";
import { GP_CAPTION, GP_MS, GaussianProcessFigure } from "@/shared/components/explainers/figures/gaussian-process";
import {
  MESSAGE_PASSING_CAPTION,
  MESSAGE_PASSING_MS,
  MessagePassingFigure,
} from "@/shared/components/explainers/figures/message-passing";
import type { ComponentType } from "react";
import { ENGINE_FIGURE, type EngineFigure } from "../types";

const FIGURES: Record<EngineFigure, { Figure: ComponentType<{ t: number }>; ms: number; caption: string }> = {
  forest: { Figure: ForestFigure, ms: FOREST_MS, caption: FOREST_CAPTION },
  boosting: { Figure: BoostingFigure, ms: BOOSTING_MS, caption: BOOSTING_CAPTION },
  "gaussian-process": { Figure: GaussianProcessFigure, ms: GP_MS, caption: GP_CAPTION },
  "message-passing": { Figure: MessagePassingFigure, ms: MESSAGE_PASSING_MS, caption: MESSAGE_PASSING_CAPTION },
  attention: { Figure: AttentionFigure, ms: ATTENTION_MS, caption: ATTENTION_CAPTION },
};

export function EngineExplainer({ engineId }: { engineId: string }) {
  const kind = ENGINE_FIGURE[engineId];
  if (!kind) return null;
  const { Figure, ms, caption } = FIGURES[kind];
  return (
    <Explainer id={`engine-${kind}`} label="How it learns" durationMs={ms} caption={caption}>
      {(t) => <Figure t={t} />}
    </Explainer>
  );
}
```

`engine-catalogue.tsx`, inside `<CardContent className="flex-1">` before the "Settings" `<p>`:
```tsx
<div className="mb-4">
  <EngineExplainer engineId={engine.id} />
</div>
```

`train-protocol-form.tsx`, directly after the Engine `</Select>` (inside the same `space-y-1.5` div):
```tsx
{engine && <EngineExplainer engineId={engine.id} />}
```

- [ ] **Step 4: Run.** `pnpm test engine-explainer train-protocol-form` → PASS (the existing form tests must stay green).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/engines frontend/src/features/protocols/components/train-protocol-form.tsx
git commit -m "feat(engines): 'How it learns' figure on engine cards and the engine picker"
```

---

### Task 5: Split, bootstrap and domain placements (B, C, D)

**Files:**
- Modify: `features/datasets/components/dataset-wizard.tsx` (step 3, after the options grid's closing `</div></div>`, before the seed field)
- Modify: `features/datasets/components/dataset-profile-view.tsx` (`SplitHonestySection`, end of `CardContent`)
- Modify: `features/protocols/components/scorecard-view.tsx` (`VerdictBand`, after the baseline-description `<p>` in the comparison branch)
- Modify: `features/protocols/components/scorecard-diagnostics.tsx` (`ApplicabilitySection`, first child of `CardContent`)
- Modify: `features/runs/components/run-detail.tsx` (before `<TriageGrid`)
- Test: `features/protocols/components/scorecard-view.test.ts`

**Interfaces:**
- Consumes: `Explainer`, `SplitFigure`, `splitCaption`, `SPLIT_MS`, `BootstrapExplainer`, `DomainFigure`, `domainCaption`, `DOMAIN_MS`, `IN_DOMAIN_FLOOR` (`features/runs/lib/result-query.ts`), `Verdict` (`features/protocols/lib/verdict.ts`).
- Produces: `showsBootstrapExplainer(verdict: Verdict): boolean`, exported from `scorecard-view.tsx`.

- [ ] **Step 1: Failing test** (`scorecard-view.test.ts`). The bootstrap explainer is gated by a pure function, so the gate is tested without rendering the whole scorecard.

```ts
import { describe, expect, it } from "vitest";
import type { Verdict } from "../lib/verdict";
import { showsBootstrapExplainer } from "./scorecard-view";

const v = (kind: Verdict["kind"], ci: [number, number] | null): Verdict => ({
  kind,
  headline: "",
  model: 0.7,
  baseline: 0.6,
  delta: 0.1,
  ci,
});

describe("showsBootstrapExplainer", () => {
  it("shows for a comparison with an interval", () => {
    expect(showsBootstrapExplainer(v("beats", [0.6, 0.8]))).toBe(true);
    expect(showsBootstrapExplainer(v("within-noise", [0.6, 0.8]))).toBe(true);
    expect(showsBootstrapExplainer(v("no-better", [0.6, 0.8]))).toBe(true);
  });
  it("hides without an interval or without a comparison", () => {
    expect(showsBootstrapExplainer(v("beats", null))).toBe(false);
    expect(showsBootstrapExplainer(v("is-baseline", [0.6, 0.8]))).toBe(false);
    expect(showsBootstrapExplainer(v("unknown", [0.6, 0.8]))).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify failure.** `pnpm test scorecard-view` fails, export missing.

- [ ] **Step 3: Implement the five placements**

`dataset-wizard.tsx`, step 3:
```tsx
<Explainer id="split" durationMs={SPLIT_MS} caption={splitCaption(draft.strategy)} replayKey={draft.strategy}>
  {(t) => <SplitFigure t={t} strategy={draft.strategy} />}
</Explainer>
```

`dataset-profile-view.tsx`, `SplitHonestySection`, last child of `CardContent`:
```tsx
<Explainer id="split" durationMs={SPLIT_MS} caption={splitCaption(isScaffoldSplit ? "scaffold" : "random")}>
  {(t) => <SplitFigure t={t} strategy={isScaffoldSplit ? "scaffold" : "random"} />}
</Explainer>
```

`scorecard-view.tsx`:
```tsx
/** The interval explainer needs an interval and a real comparison to explain. */
export function showsBootstrapExplainer(verdict: Verdict): boolean {
  return verdict.ci != null && verdict.kind !== "is-baseline" && verdict.kind !== "unknown";
}
```
and in `VerdictBand`, after the baseline-description `<p>`:
```tsx
{showsBootstrapExplainer(verdict) && (
  <div className="mt-3">
    <BootstrapExplainer startOutside={verdict.kind === "beats"} />
  </div>
)}
```
(`Verdict` is imported as a type from `../lib/verdict`.)

`scorecard-diagnostics.tsx`, `ApplicabilitySection`, first child of `CardContent`:
```tsx
<Explainer id="domain" durationMs={DOMAIN_MS} caption={domainCaption("diagnostics")}>
  {(t) => <DomainFigure t={t} threshold={IN_DOMAIN_FLOOR} />}
</Explainer>
```

`run-detail.tsx`, before `<TriageGrid`, inside the same conditional (wrap both in a fragment):
```tsx
<Explainer
  id="domain"
  label="How to read uncertainty and applicability"
  durationMs={DOMAIN_MS}
  caption={domainCaption("triage")}
>
  {(t) => <DomainFigure t={t} threshold={IN_DOMAIN_FLOOR} />}
</Explainer>
```

- [ ] **Step 4: Run.** `cd frontend && pnpm test` → whole suite PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features
git commit -m "feat(explainers): split, bootstrap-interval and applicability-domain figures in place"
```

---

### Task 6: Colored navigation icons (G)

**Files:**
- Create: `frontend/src/shared/components/icons/nav-icons.tsx`, `nav-icons.test.tsx`
- Modify: `frontend/src/app/globals.css` (append hue tokens)
- Modify: `frontend/src/shared/lib/navigation.ts`, `shared/components/layout/nav-main.tsx:33`, `shared/components/layout/command-palette.tsx:52`, `app/(dashboard)/page.tsx:34`

**Interfaces:**
- Produces: `DatasetsIcon`, `ProtocolsIcon`, `SweepsIcon`, `RunsIcon`, `CollectionsIcon`, `EnginesIcon`, `RunnersIcon`, each `(props: SVGProps<SVGSVGElement>) => ReactElement`. `NavItem.icon: ComponentType<{ className?: string }>` and `NavItem.iconClassName: string`.

- [ ] **Step 1: Failing test** (`nav-icons.test.tsx`)

```tsx
import { navigation } from "@/shared/lib/navigation";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

describe("navigation icons", () => {
  const items = navigation.flatMap((g) => g.items);
  it.each(items.map((i) => [i.title, i] as const))("%s renders a hidden, hued svg", (_, item) => {
    const Icon = item.icon;
    const { container } = render(<Icon className={item.iconClassName} />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("aria-hidden")).toBe("true");
    expect(item.iconClassName).toMatch(/^text-icon-/);
  });
});
```

- [ ] **Step 2: Run to verify failure.** `pnpm test nav-icons` fails (`iconClassName` is undefined).

- [ ] **Step 3: Implement**

`nav-icons.tsx`. Glyphs are the preview's `ICONS` (lines 1180–1188) with the approved color classes dropped; color comes from `currentColor`:
```tsx
import type { SVGProps } from "react";

/**
 * Navigation glyphs on lucide's 24-unit grid and 2-unit stroke, so they sit
 * beside the lucide icons that remain on buttons. One vocabulary: a filled
 * dot is a measured value, an open ring is a prediction, as in the logo.
 */
function Glyph({ children, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}
const Dot = (p: { cx: number; cy: number; r: number }) => <circle {...p} fill="currentColor" stroke="none" />;

export const DatasetsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3.5" y="4" width="17" height="16" rx="2" />
    <path d="M3.5 9.5h17M3.5 14.5h17M11 12h6M11 17.2h4" />
    <Dot cx={7.4} cy={12} r={1.35} />
    <Dot cx={7.4} cy={17.2} r={1.35} />
  </Glyph>
);
export const ProtocolsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M12 2.8 19.9 7.4v9.2L12 21.2l-7.9-4.6V7.4z" />
    <path d="m8.6 12.2 2.4 2.4 4.5-4.8" />
  </Glyph>
);
export const SweepsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <Dot cx={4.5} cy={12} r={1.9} />
    <path d="M6.3 11.1 17 5.6M6.6 12h10.2M6.3 12.9 17 18.4" />
    <circle cx="19" cy="4.6" r="2" />
    <circle cx="19" cy="12" r="2" />
    <circle cx="19" cy="19.4" r="2" />
  </Glyph>
);
export const RunsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M4.5 5v14l9-7z" />
    <circle cx="19" cy="8" r="2.2" />
    <circle cx="19" cy="16" r="2.2" />
  </Glyph>
);
export const CollectionsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3" y="3" width="18" height="18" rx="4" strokeDasharray="3.2 2.6" />
    <Dot cx={9} cy={9.5} r={1.5} />
    <Dot cx={15} cy={10.2} r={1.5} />
    <Dot cx={11.4} cy={15} r={1.5} />
  </Glyph>
);
export const EnginesIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M3.5 3.5v17h17" />
    <path d="M7 17c4.5 0 6.5-10 13-10.5" />
    <Dot cx={9.6} cy={13.4} r={1.4} />
    <Dot cx={14.8} cy={13.4} r={1.4} />
    <Dot cx={17.6} cy={3.8} r={1.4} />
  </Glyph>
);
export const RunnersIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3" y="4" width="18" height="7" rx="2" />
    <rect x="3" y="13" width="18" height="7" rx="2" />
    <Dot cx={7} cy={7.5} r={1.3} />
    <Dot cx={7} cy={16.5} r={1.3} />
    <path d="M11 7.5h6M11 16.5h3" />
  </Glyph>
);
```

`globals.css` (append):
```css
/* ── Navigation icon hues ─────────────────────────────────── */
/* One hue per section. Measured against --sidebar (#f1f5f9) in light mode,
   each reaches at least 3:1, including on the active row. The token amber
   (2.9:1) does not, so Runs has its own. */
:root {
  --icon-dashboard: var(--ds-text-secondary);
  --icon-datasets: var(--chart-1);
  --icon-protocols: var(--ds-success);
  --icon-sweeps: var(--chart-3);
  --icon-runs: #b45309;
  --icon-collections: #db2777;
  --icon-engines: var(--ds-score-good);
  --icon-runners: #0891b2;
}
[data-theme="dark"] {
  --icon-runs: #fbbf24;
  --icon-collections: #f472b6;
  --icon-runners: #22d3ee;
}
@theme inline {
  --color-icon-dashboard: var(--icon-dashboard);
  --color-icon-datasets: var(--icon-datasets);
  --color-icon-protocols: var(--icon-protocols);
  --color-icon-sweeps: var(--icon-sweeps);
  --color-icon-runs: var(--icon-runs);
  --color-icon-collections: var(--icon-collections);
  --color-icon-engines: var(--icon-engines);
  --color-icon-runners: var(--icon-runners);
}
```

`navigation.ts`: replace the lucide imports except `LayoutDashboard`, widen the type, and add hues:
```ts
import {
  CollectionsIcon,
  DatasetsIcon,
  EnginesIcon,
  ProtocolsIcon,
  RunnersIcon,
  RunsIcon,
  SweepsIcon,
} from "@/shared/components/icons/nav-icons";
import { LayoutDashboard } from "lucide-react";
import type { ComponentType } from "react";

export interface NavItem {
  title: string;
  href: string;
  icon: ComponentType<{ className?: string }>;
  /** The item's hue, a `text-icon-*` utility. Applied to the icon only; labels stay neutral. */
  iconClassName: string;
  children?: NavItem[];
}
```
and each item gains `iconClassName`: Dashboard `text-icon-dashboard`, Datasets `text-icon-datasets`, Protocols `text-icon-protocols`, Sweeps `text-icon-sweeps`, Runs `text-icon-runs`, Collections `text-icon-collections`, Engines `text-icon-engines`, Runners `text-icon-runners`. Icons: `DatasetsIcon`, `ProtocolsIcon`, `SweepsIcon`, `RunsIcon`, `CollectionsIcon`, `EnginesIcon`, `RunnersIcon`.

Consumers:
- `nav-main.tsx:33`: `<item.icon className={item.iconClassName} />`
- `command-palette.tsx:52`: `<item.icon className={cn("mr-2 size-4", item.iconClassName)} />` (import `cn` from `@/shared/lib/utils` if it is not already imported)
- `app/(dashboard)/page.tsx:34`: `<item.icon className={cn("size-5 shrink-0", item.iconClassName)} />`

- [ ] **Step 4: Run.** `pnpm test nav-icons && pnpm exec tsc --noEmit` → PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/shared/components/icons frontend/src/shared/lib/navigation.ts frontend/src/shared/components/layout frontend/src/app
git commit -m "feat(nav): colored navigation glyphs in the measured/predicted vocabulary"
```

---

### Task 7: Verification

- [ ] **Step 1: Full gates.** From `frontend/`: `pnpm lint && pnpm exec tsc --noEmit && pnpm test && pnpm build`. Expected: all green. Fix anything that is not, then re-run.
- [ ] **Step 2: No literal colors in figures.** `grep -rnE '#[0-9a-fA-F]{3,6}\b' frontend/src/shared/components/explainers` must print nothing.
- [ ] **Step 3: Live check.** With `make dev` running and a real sign-in, visit Engines, Train a protocol (pick each engine), New dataset step 4 (toggle the split options), a dataset page, a protocol scorecard (verdict + diagnostics), and a ready prediction run. At each one, confirm it plays once, replays on hover, stays closed after a reload once closed, follows the theme, and shows only the final frame with reduced motion emulated. Check the sidebar icons in both themes. Take a screenshot per placement.
- [ ] **Step 4: Commit any fixes** with messages that name what the live check caught.
