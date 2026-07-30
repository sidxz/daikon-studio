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
  roc_auc: "ROC AUC",
  pr_auc: "PR AUC",
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  accuracy: "Accuracy",
};

export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] ?? metric;
}
