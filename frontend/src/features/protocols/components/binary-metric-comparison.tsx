"use client";

import { ReadoutValue } from "@/shared/components/readout-value";
import { Badge } from "@/shared/components/ui/badge";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { formatCutoff } from "../lib/format-cutoff";
import { computeVerdict } from "../lib/verdict";

const VERDICT_STYLE = {
  beats: {
    label: "Ahead of baseline",
    badge: "success",
    border: "border-success/40",
    text: "text-success",
  },
  "within-noise": {
    label: "Ahead, but uncertain",
    badge: "warning",
    border: "border-warning/50",
    text: "text-warning",
  },
  ties: {
    label: "Matches baseline",
    badge: "warning",
    border: "border-warning/50",
    text: "text-muted-foreground",
  },
  "no-better": {
    label: "Behind baseline",
    badge: "warning",
    border: "border-warning/50",
    text: "text-warning",
  },
  "is-baseline": {
    label: "This is the baseline",
    badge: "outline",
    border: "border-border",
    text: "text-muted-foreground",
  },
  unknown: {
    label: "Not comparable",
    badge: "outline",
    border: "border-border",
    text: "text-muted-foreground",
  },
} as const;

const METRICS = [
  {
    key: "mcc",
    label: "MCC",
    description: "How well the model identifies active and inactive compounds at the saved cutoff.",
  },
  {
    key: "auprc",
    label: "PR AUC",
    description: "How well active compounds rise to the top of the ranking across cutoffs.",
  },
] as const;

export function BinaryMetricComparison({ scorecard }: { scorecard: ScorecardResponse }) {
  const summary = scorecard.classification_summary;
  // Use full-test counts. The scatter may be sampled and omit an entire class.
  const actives = summary ? summary.true_positive + summary.false_negative : null;
  const inactives = summary ? summary.true_negative + summary.false_positive : null;
  const total = actives != null && inactives != null ? actives + inactives : null;
  const unavailable =
    total === 0
      ? "No test compounds are available to evaluate these scores."
      : total != null && (actives === 0 || inactives === 0)
        ? `This test set contains only ${actives === 0 ? "inactive" : "active"} compounds. MCC and PR AUC cannot assess how well the model separates active and inactive compounds.`
        : null;

  return (
    <div className="mt-4 space-y-3">
      {unavailable && <p className="rounded-md border bg-muted/40 p-3 text-sm">{unavailable}</p>}
      <div className="grid gap-3 md:grid-cols-2">
        {METRICS.map(({ key, label, description }) => {
          const model = unavailable ? null : (scorecard.metrics[key] ?? null);
          // Null when the run fitted no baseline at all, which the delta below
          // already treats as "no comparison" rather than as a tie.
          const baseline = unavailable ? null : (scorecard.baseline_metrics?.[key] ?? null);
          const delta =
            !scorecard.baseline_is_self && model != null && baseline != null
              ? model - baseline
              : null;
          // Only the primary metric has a saved interval. Never present an MCC
          // interval as uncertainty on PR AUC.
          const interval =
            model != null && scorecard.primary_metric === key ? scorecard.primary_metric_ci : null;
          const verdict = computeVerdict({
            ...scorecard,
            primary_metric: key,
            primary_metric_ci: interval,
            metrics: { [key]: model },
            baseline_metrics: { [key]: baseline },
            noise_floor: null,
          });
          const style = VERDICT_STYLE[verdict.kind];
          return (
            <section
              key={key}
              aria-label={`${label} comparison`}
              className={cn("min-w-0 rounded-lg border bg-card p-4", style.border)}
            >
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="font-semibold">{label}</h3>
                <span className="text-xs text-muted-foreground">Higher is better</span>
              </div>
              <Badge variant={style.badge} className="mt-2">
                {style.label}
              </Badge>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</p>
              <dl className="mt-4 grid grid-cols-3 gap-2">
                <div>
                  <dt className="text-xs text-muted-foreground">This model</dt>
                  <dd className="mt-1 text-xl font-semibold">
                    <ReadoutValue value={model} />
                  </dd>
                </div>
                {!scorecard.baseline_is_self && (
                  <>
                    <div>
                      <dt className="text-xs text-muted-foreground">Comparison</dt>
                      <dd className="mt-1 text-xl">
                        <ReadoutValue value={baseline} />
                      </dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted-foreground">Difference</dt>
                      <dd className={cn("mt-1 text-xl tabular-nums", style.text)}>
                        {delta == null ? "N/A" : `${delta > 0 ? "+" : ""}${delta.toFixed(3)}`}
                      </dd>
                    </div>
                  </>
                )}
              </dl>
              {delta != null && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {delta === 0
                    ? "Same score as the comparison model."
                    : `${label} is ${delta > 0 ? "higher" : "lower"} than the comparison model on this test set.`}
                </p>
              )}
              {!unavailable && model == null && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {scorecard.metrics_undefined?.[key] ??
                    `${label} is not available for this model.`}
                </p>
              )}
              {!unavailable &&
                !scorecard.baseline_is_self &&
                baseline == null &&
                // Only when a baseline ran and this metric was missing from it.
                // Without one there is no comparison model whose metric could be
                // unavailable, and saying so asserts a fit that never happened.
                scorecard.baseline_metrics != null && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {label} is not available for the comparison model.
                  </p>
                )}
              {interval && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Likely range for this {label}: <ReadoutValue value={interval[0]} /> to{" "}
                  <ReadoutValue value={interval[1]} /> (95% interval for this model).
                </p>
              )}
              {verdict.kind === "within-noise" && (
                <p className="mt-2 text-xs text-warning">
                  The comparison score falls inside this model’s likely range. The observed lead is
                  uncertain.
                </p>
              )}
              <p className="mt-3 border-t pt-3 text-xs leading-relaxed text-muted-foreground">
                {key === "mcc" ? (
                  <>
                    1 is perfect; 0 means no correlation. This model uses cutoff{" "}
                    {formatCutoff(scorecard.cutoff ?? 0.5)}
                    {!scorecard.baseline_is_self &&
                      `; the comparison uses ${formatCutoff(scorecard.baseline_cutoff ?? 0.5)}`}
                    .
                  </>
                ) : (
                  <>
                    Calculated as average precision. 1 means perfect ranking.
                    {total != null && total > 0 && actives != null
                      ? ` ${actives.toLocaleString()} of ${total.toLocaleString()} test compounds are active (${((100 * actives) / total).toFixed(1)}%). Interpret PR AUC alongside this share.`
                      : " The share of active test compounds is unavailable."}
                  </>
                )}
              </p>
            </section>
          );
        })}
      </div>
    </div>
  );
}
