import type { SweepRun } from "../types";

/**
 * Metrics where a smaller number is a better model. The server picks the
 * primary metric (`primary_metric_for` in build_scorecard.py) and sends its
 * name on the row, so this table only has to know the direction -- inferring
 * it from the value would have no way to tell 0.4 RMSE from 0.4 MCC.
 */
const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

/**
 * Best first; anything unrankable last, in submission order.
 *
 * Unrankable is not the same as bad: a run still training has no number yet,
 * and a run whose metric is genuinely undefined (a single-class test split
 * makes every classification metric meaningless) has none either. Sorting
 * those as zero would rank a pending run above a real one on an RMSE sweep
 * and below it on an MCC sweep, which is a ranking that says nothing true.
 */
export function rankRuns(runs: SweepRun[]): SweepRun[] {
  const scored = runs.filter((run) => typeof run.metrics?.value === "number");
  const unscored = runs.filter((run) => typeof run.metrics?.value !== "number");
  scored.sort((a, b) => {
    const lower = LOWER_IS_BETTER.has(String(a.metrics?.primary_metric ?? ""));
    const left = a.metrics?.value as number;
    const right = b.metrics?.value as number;
    return lower ? left - right : right - left;
  });
  return [...scored, ...unscored];
}

/** The headline number, or why there isn't one. */
export function formatMetric(metrics: SweepRun["metrics"]): string {
  if (typeof metrics?.value !== "number") return "—";
  return `${String(metrics.primary_metric).toUpperCase()} ${metrics.value.toFixed(3)}`;
}

/**
 * How far this run beat its own baseline, signed so that positive always means
 * better regardless of the metric's direction. Null when either side is
 * missing -- an unmeasured comparison must not render as a dead heat.
 */
export function baselineDelta(metrics: SweepRun["metrics"]): number | null {
  const value = metrics?.value;
  const baseline = metrics?.baseline_value;
  if (typeof value !== "number" || typeof baseline !== "number") return null;
  return LOWER_IS_BETTER.has(String(metrics?.primary_metric ?? ""))
    ? baseline - value
    : value - baseline;
}
