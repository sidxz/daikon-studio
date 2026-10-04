import type { ReadoutResponse } from "@/shared/lib/api/model";

/** The cutoff a binary target is called at when none was tuned. */
export const DEFAULT_CUTOFF = 0.5;

/**
 * The largest spread a probability can have across an ensemble's models: half its
 * range, reached when they split evenly between 0 and 1. A fixed scale, so one
 * run's bars mean the same as another's.
 */
export const PROBABILITY_SPREAD_MAX = 0.5;

/**
 * Where `value` sits between `min` and `max`, clamped to 0..1, or null when there is
 * no scale to draw against (an unknown range). A range of one value fills the bar.
 */
export function positionIn(
  value: number,
  min: number | null | undefined,
  max: number | null | undefined,
): number | null {
  if (min == null || max == null || !Number.isFinite(value)) return null;
  if (max <= min) return 1;
  return Math.min(Math.max((value - min) / (max - min), 0), 1);
}

/**
 * A probability or an uncertainty to three decimals. Below 0.001 it reads "<0.001":
 * scientific notation suits a molar concentration, but a probability of 3.96e-4 or
 * a spread of 3.29e-5 is simply "almost none", and printing it differently from its
 * neighbours makes it look like an error.
 */
export function formatThousandths(value: number): string {
  if (value > 0 && value < 0.001) return "<0.001";
  return value.toFixed(3);
}

/**
 * The cutoff a probability readout's target is called at: the tuned threshold the
 * Protocol stores on that target's class readout, or 0.5 when none was tuned. The
 * pairing is by name, as the backend derives it: `{column}_probability` beside
 * `{column}`.
 */
export function cutoffFor(probability: string, readouts: ReadoutResponse[]): number {
  const column = probability.replace(/_probability$/, "");
  const threshold = readouts.find(
    (readout) => readout.name === column && readout.type === "class",
  )?.threshold;
  return typeof threshold === "number" ? threshold : DEFAULT_CUTOFF;
}
