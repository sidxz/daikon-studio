import type { EpochResponse } from "@/shared/lib/api/model";

/**
 * One neural fit's epochs: a model of a training run, as its page charts it.
 *
 * A training run fits the chosen engine, its baseline and (on a scaffold split) the
 * engine again on a random split; within each, one model per target unless the engine
 * fits every target at once, and one per ensemble member. Each of those is a series.
 */
export interface EpochSeries {
  key: string;
  label: string;
  points: EpochResponse[];
}

/** The training run's stages, as the run page names them. */
const STAGES: Record<string, string> = {
  baseline: "Baseline",
  "random-split": "Random-split comparison",
};

/** Validation scores, in the order and words the rest of the app uses. */
export const SCORE_LABELS: Record<string, string> = {
  auroc: "AUROC",
  auprc: "PR AUC",
  mcc: "MCC",
  rmse: "RMSE",
  mae: "MAE",
  r2: "R²",
};

export function seriesLabel(point: EpochResponse): string {
  const parts = [
    STAGES[point.fit],
    point.target ?? undefined,
    point.member && point.members ? `model ${point.member} of ${point.members}` : undefined,
  ].filter(Boolean);
  // No target only ever means an engine fitting every target at once (a per-target fit
  // is always tagged), so the bare model is named for what it is.
  return parts.length > 0 ? parts.join(" · ") : "One model for all targets";
}

/** Series in the order they started, each with its epochs in order. */
export function groupSeries(points: EpochResponse[]): EpochSeries[] {
  const byKey = new Map<string, EpochSeries>();
  for (const point of points) {
    const key = `${point.fit}|${point.target ?? ""}|${point.member ?? ""}`;
    let series = byKey.get(key);
    if (!series) {
      series = { key, label: seriesLabel(point), points: [] };
      byKey.set(key, series);
    }
    series.points.push(point);
  }
  for (const series of byKey.values()) series.points.sort((a, b) => a.epoch - b.epoch);
  return [...byKey.values()];
}

/** How each `kept_by` rule reads beside the kept epoch. */
const KEPT_BY: Record<string, string> = {
  auprc: "best validation PR AUC",
  auroc: "best validation AUROC",
  loss: "lowest validation loss",
};

/**
 * The epoch the fit keeps, as the fit itself last reported it (`kept_epoch`). A fit
 * recorded before fits reported it kept the lowest validation loss, so that is worked
 * out here. Null when the fit has no validation set, or when the
 * kept epoch is from an earlier attempt that this one resumed.
 */
export function keptEpoch(series: EpochSeries): EpochResponse | null {
  const reported = series.points[series.points.length - 1]?.kept_epoch;
  if (reported != null) return series.points.find((point) => point.epoch === reported) ?? null;
  let best: EpochResponse | null = null;
  for (const point of series.points) {
    if (point.val_loss === null) continue;
    if (best === null || point.val_loss < (best.val_loss as number)) best = point;
  }
  return best;
}

/** What chose the kept epoch, in words. */
export function keptRule(series: EpochSeries): string {
  const rule = series.points[series.points.length - 1]?.kept_by;
  return KEPT_BY[rule ?? "loss"] ?? rule ?? KEPT_BY.loss;
}

/** Mean seconds per epoch over the last few, or null with fewer than two epochs. */
export function secondsPerEpoch(series: EpochSeries, window = 10): number | null {
  const recent = series.points.slice(-(window + 1));
  if (recent.length < 2) return null;
  const first = Date.parse(recent[0].at);
  const last = Date.parse(recent[recent.length - 1].at);
  const seconds = (last - first) / 1000 / (recent[recent.length - 1].epoch - recent[0].epoch);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

/** Seconds until the series reaches its last epoch, at its recent pace. */
export function secondsLeft(series: EpochSeries): number | null {
  const pace = secondsPerEpoch(series);
  const last = series.points[series.points.length - 1];
  if (pace === null || !last) return null;
  return Math.max(0, last.epochs - last.epoch) * pace;
}

/** "40 s", "14 min", "2 h 5 min": a duration at the precision anyone plans with. */
export function durationLabel(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** The scores these epochs carry, in vocabulary order. */
export function scoreNames(points: EpochResponse[]): string[] {
  const present = new Set(points.flatMap((point) => Object.keys(point.scores)));
  return Object.keys(SCORE_LABELS).filter((name) => present.has(name));
}

/** Higher is better for every score except the two error measures. */
export function higherIsBetter(name: string): boolean {
  return name !== "rmse" && name !== "mae";
}
