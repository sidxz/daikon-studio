import type { ScorecardResponse } from "@/shared/lib/api/model";
import { metricLabel } from "../types";
import { higherIsBetter } from "./verdict";

/** One target's score on the metric the targets share. */
export interface TargetScore {
  target: string;
  value: number;
  baseline: number | null;
}

/** A target the average leaves out, and the reason it has no score. */
export interface SkippedTarget {
  target: string;
  reason: string;
}

export type Aggregate =
  | {
      kind: "shown";
      metric: string;
      /** Set only for a metric that carries the readout's unit; `null` otherwise. */
      unit: string | null;
      mean: number;
      /** The comparison model over the same targets, or `null` if any lacks one. */
      baselineMean: number | null;
      /** Weakest target first: the one worth looking at. */
      scored: TargetScore[];
      skipped: SkippedTarget[];
    }
  | { kind: "unavailable"; message: string };

const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

const read = (metrics: unknown, metric: string): number | null =>
  ((metrics ?? {}) as Record<string, number | null>)[metric] ?? null;

/** Blank and absent are one thing: no unit. */
const unitOf = (card: ScorecardResponse) => (card.unit?.trim() ? card.unit.trim() : null);

const list = (items: string[]) =>
  new Intl.ListFormat("en", { style: "long", type: "conjunction" }).format(items);

/**
 * The one number a multi-task benchmark reports: each target's own score,
 * averaged with every target counting once.
 *
 * Unweighted on purpose. Published multi-task results -- Tox21's mean ROC AUC
 * over twelve endpoints, a Polaris macro MAE -- weight every endpoint equally
 * regardless of how many compounds carry a label for it, so weighting by test
 * rows here would quietly disagree with the number a reader is checking against.
 *
 * Computed in the browser from the array the scorecard already holds. There is
 * no aggregate endpoint and none is needed: every per-target score has already
 * been fetched to draw the tabs.
 *
 * Refuses rather than averaging what does not belong in one number, because a
 * mean of a ROC AUC and an MAE, or of a nanomolar error and a log-unit error,
 * is a plausible-looking figure with no meaning -- the quiet kind of wrongness
 * this product exists to surface in other people's work.
 */
export function aggregateTargets(scorecards: ScorecardResponse[]): Aggregate {
  if (scorecards.length < 2)
    return {
      kind: "unavailable",
      message: "An average needs more than one target.",
    };

  const metrics = [...new Set(scorecards.map((card) => card.primary_metric))];
  if (metrics.length > 1)
    return {
      kind: "unavailable",
      message: `These targets were scored on different metrics (${list(
        metrics.map(metricLabel),
      )}), which cannot be averaged into one number.`,
    };

  const metric = metrics[0];
  const scored: TargetScore[] = [];
  const skipped: SkippedTarget[] = [];
  // Units are collected from the scored targets only, as they are walked: a
  // target with no score cannot disagree about the unit of a number it does
  // not have.
  const units = new Set<string | null>();
  for (const card of scorecards) {
    const value = read(card.metrics, metric);
    if (value == null) {
      skipped.push({
        target: card.target,
        reason:
          read(card.metrics_undefined, metric) != null
            ? String((card.metrics_undefined as Record<string, string>)[metric])
            : `${metricLabel(metric)} could not be measured for this target.`,
      });
      continue;
    }
    // The comparison model is read from the scored targets only. Averaging the
    // model over two endpoints and the baseline over three would print a
    // difference that was never measured on the same endpoints.
    scored.push({ target: card.target, value, baseline: read(card.baseline_metrics, metric) });
    units.add(unitOf(card));
  }

  if (scored.length < 2)
    return {
      kind: "unavailable",
      message:
        scored.length === 1
          ? "Only one target was scored, so there is nothing to average across."
          : "No target was scored, so there is nothing to average.",
    };

  // Whether the metric carries the readout's unit is already settled elsewhere
  // in this feature: the "higher is better" metrics are ratios and rankings, and
  // a binary endpoint cut from a micromolar readout still carries that unit on
  // its scorecard. See the same rule in `MetricTable` and `SubsetMetric`.
  const carriesUnit = !higherIsBetter(metric);
  if (carriesUnit && units.size > 1)
    return {
      kind: "unavailable",
      message: `${metricLabel(metric)} is measured in different units across these targets (${list(
        [...units].map((unit) => unit ?? "no unit"),
      )}), which cannot be averaged into one number.`,
    };

  return {
    kind: "shown",
    metric,
    unit: carriesUnit ? ([...units][0] ?? null) : null,
    mean: average(scored.map((entry) => entry.value)),
    baselineMean: scored.every((entry) => entry.baseline != null)
      ? average(scored.map((entry) => entry.baseline as number))
      : null,
    scored: scored.sort((a, b) => (higherIsBetter(metric) ? a.value - b.value : b.value - a.value)),
    skipped,
  };
}
