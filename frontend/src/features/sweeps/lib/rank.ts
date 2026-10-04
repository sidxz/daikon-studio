import type { SweepRun } from "../types";

/**
 * Metrics where a smaller number is a better model. The server picks each
 * target's primary metric (`primary_metric_for` in build_scorecard.py) and sends
 * its name on the row, so this table only has to know the direction -- inferring
 * it from the value would have no way to tell 0.4 RMSE from 0.4 MCC.
 */
const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

/** One target's headline on a training run (`Run.record_metrics`). */
export interface Headline {
  column: string;
  primary_metric: string;
  value: number | null;
  baseline_value: number | null;
}

/** A run's per-target headlines; empty while it has none. `metrics` is untyped in the contract. */
export function headlines(metrics: SweepRun["metrics"]): Headline[] {
  const targets = (metrics as { targets?: unknown } | null)?.targets;
  return Array.isArray(targets) ? (targets as Headline[]) : [];
}

/**
 * Every target any member reports, in the order reported. A sweep has one
 * dataset, so every finished member lists the same targets in the same order;
 * a pending one lists none.
 */
export function sweepTargets(runs: SweepRun[]): string[] {
  const seen: string[] = [];
  for (const run of runs) {
    for (const headline of headlines(run.metrics)) {
      if (!seen.includes(headline.column)) seen.push(headline.column);
    }
  }
  return seen;
}

export function headlineFor(run: SweepRun, column: string): Headline | undefined {
  return headlines(run.metrics).find((headline) => headline.column === column);
}

/**
 * A headline is rankable exactly when its value is a real number. Exported so
 * `sweep-detail.tsx`'s rank cell uses the same test `sortRuns` buckets on,
 * rather than a second predicate that happens to agree today.
 */
export function isRankable(headline: Headline | undefined): boolean {
  return typeof headline?.value === "number";
}

/**
 * Runs ordered by one target's headline, best first; anything unrankable last,
 * in submission order. `null` keeps submission order: with several targets no
 * single number says which run won -- an average would invent one, the worst
 * target would bury a model that is excellent at the others -- so the table
 * picks no winner until someone picks a column.
 *
 * Unrankable is not the same as bad: a run still training has no number yet, and
 * a run whose metric is genuinely undefined (a single-class test split makes
 * every classification metric meaningless) has none either. Sorting those as
 * zero would rank a pending run above a real one on an RMSE sweep and below it
 * on an MCC sweep, which is a ranking that says nothing true.
 *
 * Reads the metric direction off the LEFT operand only -- correct only because
 * one sweep has one dataset, hence one primary metric per target for every
 * member (`SubmitSweepCommand`); a mixed-direction list would sort
 * non-transitively and silently.
 */
export function sortRuns(runs: SweepRun[], column: string | null): SweepRun[] {
  if (column === null) return runs;
  const scored = runs.filter((run) => isRankable(headlineFor(run, column)));
  const unscored = runs.filter((run) => !isRankable(headlineFor(run, column)));
  scored.sort((left, right) => {
    const a = headlineFor(left, column) as Headline;
    const b = headlineFor(right, column) as Headline;
    const difference = (a.value as number) - (b.value as number);
    return LOWER_IS_BETTER.has(a.primary_metric) ? difference : -difference;
  });
  return [...scored, ...unscored];
}

/**
 * The `aria-sort` of the column `sortRuns` ordered: best first, which is
 * ascending for an error metric and descending for a score.
 */
export function sortDirection(runs: SweepRun[], column: string): "ascending" | "descending" {
  const metric = runs.map((run) => headlineFor(run, column)).find(Boolean)?.primary_metric;
  return metric && LOWER_IS_BETTER.has(metric) ? "ascending" : "descending";
}

/** The headline number, or why there isn't one. */
export function formatMetric(headline: Headline | undefined): string {
  if (!headline || !isRankable(headline)) return "—";
  return `${headline.primary_metric.toUpperCase()} ${(headline.value as number).toFixed(3)}`;
}

/**
 * How far this run beat its own baseline on one target, signed so that positive
 * always means better regardless of the metric's direction. Null when either
 * side is missing -- an unmeasured comparison must not render as a dead heat.
 */
export function baselineDelta(headline: Headline | undefined): number | null {
  if (typeof headline?.value !== "number" || typeof headline.baseline_value !== "number") {
    return null;
  }
  return LOWER_IS_BETTER.has(headline.primary_metric)
    ? headline.baseline_value - headline.value
    : headline.value - headline.baseline_value;
}
