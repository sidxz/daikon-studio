"use client";

import { useEngines } from "@/features/engines";
import {
  type BootstrapData,
  BootstrapExplainer,
} from "@/shared/components/explainers/figures/bootstrap";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Progress } from "@/shared/components/ui/progress";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { splitVocabulary } from "@/shared/lib/split";
import { cn } from "@/shared/lib/utils";
import { useState } from "react";
import { formatCutoff } from "../lib/format-cutoff";
import {
  type Verdict,
  computeOptimismGap,
  computeVerdict,
  describeBaseline,
  higherIsBetter,
  signed,
} from "../lib/verdict";
import { METRIC_DESCRIPTIONS, metricLabel } from "../types";
import { BinaryMetricComparison } from "./binary-metric-comparison";
import { ConfusionMatrix } from "./confusion-matrix";
import { ScorecardDiagnostics, SplitComparison, SplitDrawSpread } from "./scorecard-diagnostics";
import { ScorecardOverview } from "./scorecard-overview";
import { ScorecardSection } from "./scorecard-section";
import { ScorecardSimilarity } from "./scorecard-similarity";

function HonestyStat({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-[9rem] flex-1">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

/**
 * The three numbers that argue with the verdict, inside the verdict's own
 * border.
 *
 * They were three separate cards below the band, and separate blocks can be
 * read separately: the ESOL run showed a green "beats the baseline" while an
 * optimism gap six times the winning margin sat in a card underneath it. In
 * one border, nobody reads the claim without the doubts. The captions stay
 * visible for the same reason -- hover-to-see-the-caveat is a way of hiding
 * one.
 *
 * Rendered on every verdict, including `is-baseline` and `unknown`, not just
 * the head-to-head branch: the optimism gap compares the model's own
 * random-split score to its own grouped-split score, and applicability
 * coverage compares training set to test set -- neither involves the
 * baseline, so neither has anything to do with whether a comparison exists.
 * Only the noise floor is comparison-adjacent, and it already degrades on its
 * own via the `noise_floor != null` guard below. Training the product's own
 * baseline engine (`ecfp4-randomforest`, `is-baseline`) still has an
 * optimism gap and an applicability rate worth knowing.
 */
function HonestyStats({ scorecard }: { scorecard: ScorecardResponse }) {
  const gap = computeOptimismGap(scorecard);
  const coverage = scorecard.applicability_coverage;
  const metric = metricLabel(scorecard.primary_metric);
  const vocabulary = splitVocabulary(scorecard.split_strategy);
  const group = vocabulary.group ?? "group of related rows";

  return (
    <div className="mt-4 flex flex-wrap gap-x-8 gap-y-4 border-t border-current/15 pt-3">
      <HonestyStat label={`Performance on ${vocabulary.unfamiliar}`}>
        {gap.kind === "shown" ? (
          <>
            <ReadoutValue value={gap.gap} precision={3} className="text-xl font-semibold" />
            <p className="mt-1 text-xs text-muted-foreground">
              {metric} was <ReadoutValue value={gap.random} precision={3} /> when one {group} could
              appear in both training and testing, and{" "}
              <ReadoutValue value={gap.held} precision={3} /> when each {group} was kept to one
              side. This difference is the optimism gap.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">{gap.message}</p>
        )}
      </HonestyStat>

      {/* Absent, not empty, for a binary target: there are no replicate
          spreads to average, so the question does not arise. */}
      {scorecard.noise_floor != null ? (
        <HonestyStat label="Variation in repeated measurements">
          <ReadoutValue
            value={scorecard.noise_floor}
            unit={scorecard.unit}
            precision={3}
            className="text-xl font-semibold"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Average spread when the same compound was measured more than once. This gives context
            for the model's errors.
          </p>
        </HonestyStat>
      ) : scorecard.deduplicated === false ? (
        /* Three cases, not two. A binary target has no spread to measure and a silent
           absence is right for it; a dataset built without deduplication has no
           replicate groups at all, and the same silence there reads as a missing
           number rather than as the consequence of a choice. `null` is a scorecard
           written before the toggle existed, which claims nothing. */
        <HonestyStat label="Variation in repeated measurements">
          <p className="text-sm text-muted-foreground">
            Not measured: deduplication was switched off for this dataset, so repeated measurements
            of one compound were kept as separate rows rather than compared.
          </p>
        </HonestyStat>
      ) : null}

      <HonestyStat label="Test compounds similar to training">
        {coverage == null ? (
          <p className="text-xs text-muted-foreground">Could not be computed for this protocol.</p>
        ) : (
          <>
            <span className="text-xl font-semibold tabular-nums">
              {(coverage * 100).toFixed(0)}%
            </span>
            <Progress value={coverage * 100} className="mt-1.5 h-1.5" />
            <p className="mt-1 text-xs text-muted-foreground">
              have a training compound with a structural similarity score of at least 0.3. This
              measures familiarity, not the percentage of correct predictions.
            </p>
          </>
        )}
      </HonestyStat>
    </div>
  );
}

/** The interval explainer needs an interval, its redraws, and a real comparison to explain. */
export function bootstrapData(
  scorecard: ScorecardResponse,
  verdict: Verdict,
): BootstrapData | null {
  if (verdict.baseline == null) return null;
  if (verdict.kind === "is-baseline" || verdict.kind === "unknown") return null;
  // Paired when the card is: the figure explains the test the verdict used.
  const paired =
    verdict.difference && scorecard.difference_bootstrap
      ? { interval: verdict.difference, redraws: scorecard.difference_bootstrap }
      : null;
  const interval = paired?.interval ?? verdict.ci;
  const redraws = paired?.redraws ?? scorecard.primary_metric_bootstrap;
  if (interval == null || redraws == null) return null;
  return {
    mode: paired ? "difference" : "score",
    metric: metricLabel(scorecard.primary_metric),
    higherIsBetter: higherIsBetter(scorecard.primary_metric),
    interval,
    baseline: paired ? 0 : verdict.baseline,
    redraws,
    compounds: scorecard.parity,
    testSize: scorecard.parity_sampled_from ?? scorecard.parity.length,
    cutoff: scorecard.prediction_kind === "probability" ? (scorecard.cutoff ?? 0.5) : null,
  };
}

/**
 * The decision cutoff behind the MCC, or why tuning did not happen. Absent when tuning
 * was not requested: the cutoff is then 0.5 and unremarkable.
 *
 * `cutoff` is set only when tuning was requested and succeeded for the model. The
 * baseline can still lack one when the model has one: both see the same validation set,
 * so class counts are not the cause, but the baseline's probabilities can be constant
 * or its training rows single-class. Say so, because the comparison is then tuned
 * against 0.5. `comparesBaseline` is false when the baseline is the same fit.
 */
function CutoffLine({
  scorecard,
  comparesBaseline,
}: {
  scorecard: ScorecardResponse;
  comparesBaseline: boolean;
}) {
  const { cutoff, baseline_cutoff: baselineCutoff, cutoff_note: note } = scorecard;
  const model = cutoff != null ? `At cutoff ${formatCutoff(cutoff)}, tuned on validation.` : note;
  if (!model) return null;
  const baseline = !comparesBaseline
    ? null
    : baselineCutoff != null
      ? `Baseline at its own tuned cutoff ${formatCutoff(baselineCutoff)}.`
      : cutoff != null
        ? "Baseline at 0.5: its validation predictions could not support a cutoff."
        : null;
  return (
    <p className="mt-2 text-xs text-muted-foreground">
      {model}
      {baseline && ` ${baseline}`}
    </p>
  );
}

function VerdictBand({ scorecard }: { scorecard: ScorecardResponse }) {
  const binary = scorecard.prediction_kind === "probability";
  const verdict = computeVerdict(scorecard);
  // A third state beside "beats" and "is-baseline": the run was asked not to fit one.
  // Nothing failed and nothing is undefined -- there is simply nothing to compare, and
  // every sentence on this card that names a comparison model has to know that.
  const noBaseline = scorecard.baseline_metrics == null;
  const bootstrap = bootstrapData(scorecard, verdict);
  const metric = metricLabel(scorecard.primary_metric);
  const { data: engines } = useEngines();
  const engineName =
    engines?.find((engine) => engine.id === scorecard.engine_id)?.name ?? scorecard.engine_id;

  const tone =
    verdict.kind === "beats"
      ? "border-success/40 bg-success/5"
      : verdict.kind === "no-better" || verdict.kind === "ties" || verdict.kind === "within-noise"
        ? "border-warning/50 bg-warning/5"
        : "border-border bg-muted/30";

  return (
    <section aria-label="Comparison & reliability" className={cn("rounded-lg border p-5", tone)}>
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Comparison & reliability
      </p>
      <p className="text-lg font-semibold">
        {binary && verdict.kind !== "is-baseline"
          ? verdict.kind === "unknown"
            ? `${metric} comparison unavailable`
            : `${verdict.headline} (${metric})`
          : verdict.headline}
      </p>
      {/* Which split produced these numbers -- `scorecard.py`'s own docstring
          calls this "the single most important fact about how flattering a
          number is allowed to be", and it was on the wire and unrendered. It
          goes directly under the headline because it qualifies the headline. */}
      <p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">
        scored on a {scorecard.split_strategy} split:{" "}
        {splitVocabulary(scorecard.split_strategy).held}
      </p>

      {binary ? (
        <>
          <BinaryMetricComparison scorecard={scorecard} />
          <p className="mt-3 text-xs text-muted-foreground">
            {noBaseline
              ? "No baseline was fitted for this run, so the comparison columns are empty."
              : scorecard.baseline_is_self
                ? `This model is the baseline (${engineName}). There is no separate model to compare.`
                : `The comparison model is ${describeBaseline(scorecard, engines)}, evaluated on the same test set. Differences above are observed scores, not proof that one model will perform better on new compounds.`}
          </p>
          {bootstrap && scorecard.metrics[scorecard.primary_metric] != null && (
            <div className="mt-3 space-y-2 text-xs text-muted-foreground">
              <h4 className="font-medium">How is the 95% interval estimated?</h4>
              <BootstrapExplainer data={bootstrap} />
            </div>
          )}
        </>
      ) : verdict.kind === "is-baseline" ? (
        <p className="mt-2 text-sm text-muted-foreground">
          The model and baseline are the same engine ({engineName}) with the same settings, so there
          is no comparison to report.
        </p>
      ) : noBaseline ? (
        /* Before the `unknown` arm, which is written for "the metric was undefined".
           No baseline is a different statement: nothing failed, nothing was measured,
           and saying "could not be computed" sends the reader to All metrics for a
           reason that is not there. */
        <p className="mt-2 text-sm text-muted-foreground">
          No baseline was fitted for this run, so there is no comparison to report. The model's own
          scores are below.
        </p>
      ) : verdict.kind === "unknown" ? (
        <p className="mt-2 text-sm text-muted-foreground">
          {metric} could not be computed for the model or the baseline, so no comparison is
          possible. See All metrics below for the reason.
        </p>
      ) : (
        <>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <span className="text-sm">
              <span className="text-muted-foreground">{metric}</span>{" "}
              <ReadoutValue
                value={verdict.model}
                unit={higherIsBetter(scorecard.primary_metric) ? undefined : scorecard.unit}
                className="text-xl font-semibold"
              />
            </span>
            <span className="text-sm">
              <span className="text-muted-foreground">baseline</span>{" "}
              <ReadoutValue
                value={verdict.baseline}
                unit={higherIsBetter(scorecard.primary_metric) ? undefined : scorecard.unit}
                className="text-xl font-semibold"
              />
            </span>
            {verdict.delta != null && (
              <span className="text-sm text-muted-foreground">
                {verdict.delta > 0 ? "+" : ""}
                {verdict.delta.toFixed(3)}
              </span>
            )}
          </div>
          {verdict.difference ? (
            <p className="mt-1 text-xs text-muted-foreground">
              95% interval for the difference: {signed(verdict.difference[0])} to{" "}
              {signed(verdict.difference[1])}
            </p>
          ) : (
            verdict.ci && (
              <p className="mt-1 text-xs text-muted-foreground">
                Likely range for this {metric}: <ReadoutValue value={verdict.ci[0]} /> to{" "}
                <ReadoutValue value={verdict.ci[1]} /> (95% interval)
              </p>
            )
          )}
          {/* Within noise by the interval: the only such verdict with no noise floor. */}
          {verdict.kind === "within-noise" && verdict.noiseFloor == null && (
            <p className="mt-2 text-sm">
              {verdict.difference
                ? "The interval for the difference includes zero, so the two models are not distinguishable on this test set."
                : `The baseline's ${metric} lies within this interval, so the two models are not distinguishable on this test set.`}
            </p>
          )}
          {verdict.kind === "within-noise" && verdict.noiseFloor != null && (
            <p className="mt-2 text-sm">
              The margin (
              <ReadoutValue value={Math.abs(verdict.delta ?? 0)} precision={3} />) is smaller than
              the assay noise floor (
              <ReadoutValue value={verdict.noiseFloor} unit={scorecard.unit} precision={3} />
              ). The two models are indistinguishable on this data; prefer the simpler one.
            </p>
          )}
          <p className="mt-2 text-sm text-muted-foreground">
            The baseline is {describeBaseline(scorecard, engines)} on the same dataset and the same
            split.
            {/* This claim is specifically about fingerprint baselines -- it is
                false about e.g. a chemprop baseline, and this page's whole
                purpose is to tell a scientist the truth about their model. */}
          </p>
          {bootstrap && (
            <div className="mt-3 space-y-2 text-xs text-muted-foreground">
              <h4 className="font-medium">How is the 95% interval estimated?</h4>
              <BootstrapExplainer data={bootstrap} />
            </div>
          )}
        </>
      )}

      {/* MCC uses the saved decision cutoff; PR AUC is independent of that cutoff. */}
      <CutoffLine scorecard={scorecard} comparesBaseline={verdict.kind !== "is-baseline"} />
      <HonestyStats scorecard={scorecard} />
    </section>
  );
}

/**
 * A metric over one named part of the test set.
 *
 * Its own card rather than a row in `MetricTable`, deliberately. A number with a
 * different denominator sitting unlabelled beside the full-test metrics invites the
 * reader to compare two things that were not measured on the same rows -- which is the
 * quiet kind of wrongness this product exists to surface in other people's work. The
 * count and the total are part of the heading for the same reason.
 */
function SubsetMetric({ scorecard }: { scorecard: ScorecardResponse }) {
  const column = scorecard.subset_column;
  if (column == null) return null;
  const count = scorecard.subset_count ?? 0;
  const total = scorecard.subset_total ?? 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Metrics on the rows flagged by <span className="font-mono">{column}</span>
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          {count} of {total} test compounds. Published benchmarks often report a number over part of
          their test set; this is that number, measured the same way.
        </p>
      </CardHeader>
      <CardContent>
        {scorecard.subset_metric == null ? (
          /* Three causes, not one. Nothing flagged in the test set; flagged rows that
             are all one class, where MCC is undefined; or a mask that does not describe
             these rows. Printing "no test row carries this flag" above a count of 245
             is the contradiction this branch exists to avoid. */
          <p className="text-sm text-muted-foreground">
            {count === 0
              ? "No test row carries this flag, so there is nothing to measure here."
              : `${metricLabel(scorecard.primary_metric)} is undefined on these rows: every flagged compound has the same label.`}
          </p>
        ) : (
          <dl className="flex items-baseline gap-3">
            <dt className="text-sm text-muted-foreground">
              {metricLabel(scorecard.primary_metric)}
            </dt>
            <dd>
              <ReadoutValue
                value={scorecard.subset_metric}
                /* Same rule as MetricTable: the "higher is better" metrics are
                   unitless, and a binary target carrying a unit would otherwise
                   render "MCC 0.432 nM". */
                unit={higherIsBetter(scorecard.primary_metric) ? undefined : scorecard.unit}
                precision={3}
                className="text-xl font-semibold"
              />
            </dd>
          </dl>
        )}
      </CardContent>
    </Card>
  );
}

function MetricTable({ scorecard }: { scorecard: ScorecardResponse }) {
  const metrics = (scorecard.metrics ?? {}) as Record<string, number | null>;
  const baseline = (scorecard.baseline_metrics ?? {}) as Record<string, number | null>;
  const validation = scorecard.validation_metrics as Record<string, number | null> | null;
  const undefinedReasons = (scorecard.metrics_undefined ?? {}) as Record<string, string>;
  const names = Object.keys(metrics);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">All metrics</CardTitle>
        {/* The validation column is not decoration, it is the point. Every number
            under "test" is spent the moment it is used to choose between two sets
            of conditions -- so the page has to offer somewhere else to look, and
            say plainly which one is which. */}
        {validation ? (
          <p className="text-sm text-muted-foreground">
            Tune settings against the <span className="font-medium">validation</span> column.
            Reserve the test column for the final comparison: each time it informs a choice, it
            becomes less of a held-out set and its estimate more optimistic.
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            No validation metrics: the split has no validation set, or the run predates validation
            scoring. Avoid tuning settings against the test column.
          </p>
        )}
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Metric</th>
              {validation && (
                <th className="pb-2 pr-4 font-medium">
                  Validation
                  <span className="ml-1 normal-case text-[10px]">For tuning</span>
                </th>
              )}
              <th className="pb-2 pr-4 font-medium">
                Test
                <span className="ml-1 normal-case text-[10px]">the verdict</span>
              </th>
              {/* Blank when the model is its own baseline, and equally when no
                  baseline ran -- an unexplained column of N/A is worse than no column. */}
              <th className="pb-2 font-medium">
                {scorecard.baseline_is_self || scorecard.baseline_metrics == null ? "" : "Baseline"}
              </th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const value = metrics[name];
              const reason = undefinedReasons[name];
              const isPrimary = name === scorecard.primary_metric;
              return (
                <tr key={name} className="border-b last:border-0 align-top">
                  <td className={cn("py-3 pr-4", isPrimary && "font-medium")}>
                    {metricLabel(name)}
                    {isPrimary && (
                      <span className="ml-2 text-xs font-normal text-muted-foreground">
                        primary
                      </span>
                    )}
                    <p className="mt-1 max-w-sm text-xs font-normal text-muted-foreground">
                      {METRIC_DESCRIPTIONS[name]} {higherIsBetter(name) ? "Higher" : "Lower"} is
                      better.
                    </p>
                  </td>
                  {validation && (
                    <td className="py-2 pr-4">
                      <ReadoutValue
                        value={validation[name]}
                        unit={higherIsBetter(name) ? undefined : scorecard.unit}
                      />
                    </td>
                  )}
                  <td className="py-2 pr-4">
                    {value == null && reason ? (
                      // Never a bare blank where a number belongs. The reason is
                      // written for a scientist and says what to do about it.
                      <span className="text-xs text-warning">{reason}</span>
                    ) : (
                      <ReadoutValue
                        value={value}
                        unit={higherIsBetter(name) ? undefined : scorecard.unit}
                      />
                    )}
                  </td>
                  <td className="py-2">
                    {scorecard.baseline_is_self ? (
                      <span className="text-xs text-muted-foreground">is the baseline</span>
                    ) : (
                      <ReadoutValue
                        value={baseline[name]}
                        unit={higherIsBetter(name) ? undefined : scorecard.unit}
                        className="text-muted-foreground"
                      />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

export function ScorecardView({
  scorecard,
  protocolId,
}: { scorecard: ScorecardResponse; protocolId?: string }) {
  const [tolerance, setTolerance] = useState<{ target: string; value: number | null } | null>(null);
  return (
    <div className="space-y-5">
      <ScorecardOverview
        scorecard={scorecard}
        protocolId={protocolId}
        onToleranceChange={(value) => setTolerance({ target: scorecard.target, value })}
      />
      <VerdictBand scorecard={scorecard} />
      <ConfusionMatrix scorecard={scorecard} />
      <ScorecardDiagnostics scorecard={scorecard} mode="overview" />
      <ScorecardSection
        title="Where does the model struggle?"
        description="Explore how performance changes with training similarity and chemical family."
      >
        <ScorecardSimilarity
          key={scorecard.target}
          scorecard={scorecard}
          protocolId={protocolId}
          tolerance={tolerance && tolerance.target === scorecard.target ? tolerance.value : null}
        />
        <ScorecardDiagnostics scorecard={scorecard} mode="detail" />
      </ScorecardSection>
      <ScorecardSection
        title="All metrics and comparisons"
        description={
          scorecard.baseline_metrics == null
            ? "Validation and test scores, and the effect of the data split. No baseline was fitted for this run."
            : "Validation and test scores, the comparison model, and the effect of the data split."
        }
      >
        <MetricTable scorecard={scorecard} />
        <SubsetMetric scorecard={scorecard} />
        <SplitComparison scorecard={scorecard} />
        <SplitDrawSpread scorecard={scorecard} />
      </ScorecardSection>
      <ScorecardSection
        title="Training settings"
        description="The model and comparison settings used for this protocol."
      >
        <Conditions scorecard={scorecard} />
      </ScorecardSection>
    </div>
  );
}

/**
 * The hyperparameters that produced this model, and the baseline's.
 *
 * Persisted on the Protocol since it was first written and never displayed,
 * which made a published Protocol less reproducible than the data behind it
 * already was.
 */
function Conditions({ scorecard }: { scorecard: ScorecardResponse }) {
  const conditions = (scorecard.conditions ?? {}) as Record<string, unknown>;
  const baseline = (scorecard.baseline_conditions ?? {}) as Record<string, unknown>;
  const names = [...new Set([...Object.keys(conditions), ...Object.keys(baseline)])].sort();
  if (names.length === 0)
    return <p className="text-sm text-muted-foreground">No training settings were recorded.</p>;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training settings</CardTitle>
        <p className="text-sm text-muted-foreground">
          Resolved settings for the model and baseline. Reproducing this protocol requires this
          engine, these settings and the cited dataset.
        </p>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Setting</th>
              <th className="pb-2 pr-4 font-medium">{scorecard.engine_id}</th>
              <th className="pb-2 font-medium">
                {scorecard.baseline_is_self ? "" : scorecard.baseline_engine_id}
              </th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name} className="border-b last:border-0">
                <td className="py-2 pr-4 font-mono text-xs">{name}</td>
                <td className="py-2 pr-4 tabular-nums">{format(conditions[name])}</td>
                <td className="py-2 tabular-nums text-muted-foreground">
                  {scorecard.baseline_is_self ? "N/A" : format(baseline[name])}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function format(value: unknown): string {
  if (value == null) return "N/A";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
