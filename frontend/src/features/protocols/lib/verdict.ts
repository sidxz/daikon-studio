import type { ScorecardResponse } from "@/shared/lib/api/model";

export type VerdictKind =
  | "beats"
  | "within-noise"
  | "ties"
  | "no-better"
  | "is-baseline"
  | "unknown";

export interface Verdict {
  kind: VerdictKind;
  headline: string;
  model: number | null;
  baseline: number | null;
  delta: number | null;
  /** Set when the margin was judged against the assay's own noise. */
  noiseFloor?: number | null;
}

/**
 * Metrics where a smaller number is a better model, regardless of which way the
 * *readout* points. `direction` on the dataset says whether a high IC50 is good;
 * that is a separate question from whether a high RMSE is good, and conflating
 * the two would invert the verdict on every error metric.
 */
const LOWER_IS_BETTER = new Set(["rmse", "mae", "mse", "logloss", "log_loss", "brier"]);

/** Below this the two numbers are the same model as far as a scientist cares. */
const TIE_EPSILON = 1e-9;

export function higherIsBetter(metric: string): boolean {
  return !LOWER_IS_BETTER.has(metric.toLowerCase());
}

/**
 * Is this model worth anything over fingerprints and a random forest?
 *
 * Success criterion 3 is that the answer is obvious on the first screen, so it
 * is computed once, here, rather than left to the reader to infer from two
 * numbers side by side.
 */
export function computeVerdict(scorecard: ScorecardResponse): Verdict {
  const metric = scorecard.primary_metric;
  const model = (scorecard.metrics as Record<string, number | null>)?.[metric] ?? null;
  const baseline = (scorecard.baseline_metrics as Record<string, number | null>)?.[metric] ?? null;

  if (scorecard.baseline_is_self) {
    return {
      kind: "is-baseline",
      headline: "This model is the baseline",
      model,
      baseline: null,
      delta: null,
    };
  }

  if (model == null || baseline == null) {
    return {
      kind: "unknown",
      headline: "Cannot be compared to the baseline",
      model,
      baseline,
      delta: null,
    };
  }

  const delta = model - baseline;
  const better = higherIsBetter(metric) ? delta > 0 : delta < 0;

  if (Math.abs(delta) < TIE_EPSILON) {
    return { kind: "ties", headline: "Identical to the baseline", model, baseline, delta: 0 };
  }

  // A win smaller than the assay's own measurement error is not a win. The
  // duplicate spread in the training data is a free estimate of that error,
  // and comparing the margin against it is the difference between "beats the
  // baseline" and "beats the baseline by less than you can measure".
  //
  // Only applied to error metrics, where the metric and the noise floor share
  // the readout's units and the comparison is meaningful. R² is dimensionless,
  // so holding it against a noise floor in log mol/L would be arithmetic on
  // unrelated quantities -- exactly the kind of number this product refuses to
  // manufacture elsewhere.
  const noiseFloor = scorecard.noise_floor;
  const comparableToNoise = !higherIsBetter(metric);
  if (better && comparableToNoise && noiseFloor != null && Math.abs(delta) < noiseFloor) {
    return {
      kind: "within-noise",
      headline: "Ahead of the baseline, but by less than the assay noise",
      model,
      baseline,
      delta,
      noiseFloor,
    };
  }

  return better
    ? { kind: "beats", headline: "Beats the baseline", model, baseline, delta, noiseFloor }
    : {
        kind: "no-better",
        headline: "No better than the baseline",
        model,
        baseline,
        delta,
        noiseFloor,
      };
}

/**
 * What to call the baseline in prose, given that it is now choosable.
 *
 * When the baseline is a different engine, naming it says everything a
 * scientist needs. When it is the *same* engine (comparing a pretrained
 * chemprop run against an unpretrained one, say), naming the engine again
 * would render "chemprop-dmpnn versus chemprop-dmpnn" -- true but useless.
 * What actually distinguishes the two runs is their conditions, so this
 * names whichever keys differ instead.
 */
export function describeBaseline(
  scorecard: Pick<
    ScorecardResponse,
    "baseline_engine_id" | "engine_id" | "conditions" | "baseline_conditions"
  >,
): string {
  if (scorecard.baseline_engine_id !== scorecard.engine_id) {
    return scorecard.baseline_engine_id;
  }

  const conditions = scorecard.conditions as Record<string, unknown>;
  const baselineConditions = scorecard.baseline_conditions as Record<string, unknown>;

  // A Run trained before the baseline became choosable has no recorded
  // baseline conditions at all (`field(default_factory=dict)` makes that blob
  // readable, not renderable). There is nothing to diff against, so naming the
  // engine is the only true thing left to say -- the alternative is a
  // "key = undefined" for every key `conditions` happens to have.
  if (Object.keys(baselineConditions).length === 0) return scorecard.baseline_engine_id;

  const keys = new Set([...Object.keys(conditions), ...Object.keys(baselineConditions)]);
  const differing = [...keys].filter((key) => conditions[key] !== baselineConditions[key]).sort();

  if (differing.length === 0) {
    // Same engine, same conditions -- this is `baseline_is_self`, rendered
    // through a different verdict branch entirely, but a caller that reaches
    // here anyway must still say something true rather than an empty phrase.
    return `the same engine (${scorecard.engine_id}) with the same conditions`;
  }

  const detail = differing.map((key) => `${key} = ${String(baselineConditions[key])}`).join(", ");
  return `the same engine with ${detail}`;
}

export type GapKind = "shown" | "not-applicable" | "unavailable";

export interface OptimismGap {
  kind: GapKind;
  scaffold: number | null;
  random: number | null;
  gap: number | null;
  message: string | null;
}

/**
 * How much of the score is an artefact of the split.
 *
 * The three states must read differently. Both nulls means the question does
 * not arise, because the model was already trained on a random split. A
 * non-null `random_split_unavailable` means the comparison was attempted and
 * could not be computed, and its message says why. Collapsing those two into
 * one blank is exactly the failure this product exists to prevent.
 */
export function computeOptimismGap(scorecard: ScorecardResponse): OptimismGap {
  const metric = scorecard.primary_metric;
  const scaffold = (scorecard.metrics as Record<string, number | null>)?.[metric] ?? null;
  const random =
    (scorecard.random_split_metrics as Record<string, number | null> | null)?.[metric] ?? null;

  if (scorecard.random_split_unavailable) {
    return {
      kind: "unavailable",
      scaffold,
      random: null,
      gap: null,
      message: scorecard.random_split_unavailable,
    };
  }

  if (scorecard.random_split_metrics == null) {
    return {
      kind: "not-applicable",
      scaffold,
      random: null,
      gap: null,
      message:
        "This model was trained on a random split, so there is no more optimistic split to compare it against.",
    };
  }

  if (scaffold == null || random == null) {
    return {
      kind: "unavailable",
      scaffold,
      random,
      gap: null,
      message: `The ${metric} was undefined on one of the two splits.`,
    };
  }

  // Signed so it always means "how much the easy split flattered the model",
  // whichever way the metric points.
  const gap = higherIsBetter(metric) ? random - scaffold : scaffold - random;
  return { kind: "shown", scaffold, random, gap, message: null };
}
