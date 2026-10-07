import type { ProtocolResponse, ScorecardResponse } from "@/shared/lib/api/model";

export type Protocol = ProtocolResponse;
export type Scorecard = ScorecardResponse;

/**
 * Metric names as a scientist says them. Anything not listed falls through as
 * its raw key rather than being hidden -- a new backend metric should show up
 * looking slightly raw, never vanish.
 */
export const METRIC_LABELS: Record<string, string> = {
  r2: "R²",
  rmse: "RMSE",
  mae: "MAE",
  spearman: "Spearman ρ",
  pearson: "Pearson r",
  mcc: "MCC",
  balanced_accuracy: "Balanced accuracy",
  // The keys `infrastructure/engines/_scoring.py` actually emits. This map
  // previously said `roc_auc`/`pr_auc`, which no backend metric is called, so
  // every classification scorecard rendered the raw keys instead of these
  // labels. `metricLabel`'s fallback made it a cosmetic bug rather than a
  // crash, which is exactly why it survived.
  auroc: "ROC AUC",
  auprc: "PR AUC",
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  accuracy: "Accuracy",
};

export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] ?? metric;
}

export const METRIC_DESCRIPTIONS: Record<string, string> = {
  mae: "Average distance from the measured value.",
  rmse: "Prediction error with extra weight on larger mistakes.",
  r2: "How much variation is captured. 1 is perfect; 0 matches always predicting the test-set average.",
  mcc: "Overall quality of binary decisions, accounting for both classes. 1 is perfect; 0 means no correlation.",
  balanced_accuracy:
    "Average share correctly identified within each class, giving both classes equal weight.",
  auroc: "How well active compounds rank above inactive ones. 1 is perfect; 0.5 is random ranking.",
  auprc:
    "How well active compounds are retrieved across cutoffs. Interpret alongside the share of actives in the test set.",
  precision: "Of compounds predicted active, the share that actually was active.",
  recall: "Of all truly active compounds, the share the model found.",
};
