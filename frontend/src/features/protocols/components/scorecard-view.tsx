"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Progress } from "@/shared/components/ui/progress";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { computeOptimismGap, computeVerdict, describeBaseline } from "../lib/verdict";
import { metricLabel } from "../types";
import { ScorecardDiagnostics, SplitComparison } from "./scorecard-diagnostics";

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
 * random-split score to its own scaffold-split score, and applicability
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

  return (
    <div className="mt-4 flex flex-wrap gap-x-8 gap-y-4 border-t border-current/15 pt-3">
      <HonestyStat label="Optimism gap">
        {gap.kind === "shown" ? (
          <>
            <ReadoutValue value={gap.gap} precision={3} className="text-xl font-semibold" />
            <p className="mt-1 text-xs text-muted-foreground">
              {metric} was <ReadoutValue value={gap.random} precision={3} /> on a random split and{" "}
              <ReadoutValue value={gap.scaffold} precision={3} /> on the scaffold split it was
              actually scored on — the difference an easier split would have flattered it by.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">{gap.message}</p>
        )}
      </HonestyStat>

      {/* Absent, not empty, for a binary target: there are no replicate
          spreads to average, so the question does not arise. */}
      {scorecard.noise_floor != null && (
        <HonestyStat label="Assay noise floor">
          <ReadoutValue
            value={scorecard.noise_floor}
            unit={scorecard.unit}
            precision={3}
            className="text-xl font-semibold"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Repeat measurements of the same compound disagreed by this much. No model trained on
            this data can honestly do better.
          </p>
        </HonestyStat>
      )}

      <HonestyStat label="Applicability">
        {coverage == null ? (
          <p className="text-xs text-muted-foreground">Could not be computed for this protocol.</p>
        ) : (
          <>
            <span className="text-xl font-semibold tabular-nums">
              {(coverage * 100).toFixed(0)}%
            </span>
            <Progress value={coverage * 100} className="mt-1.5 h-1.5" />
            <p className="mt-1 text-xs text-muted-foreground">
              of test compounds sit close enough to the training set for the model to have seen
              anything like them. The rest is extrapolation.
            </p>
          </>
        )}
      </HonestyStat>
    </div>
  );
}

function VerdictBand({ scorecard }: { scorecard: ScorecardResponse }) {
  const verdict = computeVerdict(scorecard);
  const metric = metricLabel(scorecard.primary_metric);

  const tone =
    verdict.kind === "beats"
      ? "border-success/40 bg-success/5"
      : verdict.kind === "no-better" || verdict.kind === "ties" || verdict.kind === "within-noise"
        ? "border-warning/50 bg-warning/5"
        : "border-border bg-muted/30";

  return (
    <div className={cn("rounded-lg border p-5", tone)}>
      <p className="text-lg font-semibold">{verdict.headline}</p>
      {/* Which split produced these numbers -- `scorecard.py`'s own docstring
          calls this "the single most important fact about how flattering a
          number is allowed to be", and it was on the wire and unrendered. It
          goes directly under the headline because it qualifies the headline. */}
      <p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">
        scored on a {scorecard.split_strategy} split
        {scorecard.split_strategy === "random"
          ? " — close analogues of training compounds are in the test set, so this reads high"
          : " — test compounds have ring systems the model never trained on"}
      </p>

      {verdict.kind === "is-baseline" ? (
        <p className="mt-2 text-sm text-muted-foreground">
          You trained {scorecard.engine_id}, which is what every other model here is measured
          against. There is nothing to compare it to — a comparison against itself would be a number
          that means nothing.
        </p>
      ) : verdict.kind === "unknown" ? (
        <p className="mt-2 text-sm text-muted-foreground">
          The {metric} could not be computed on one side of the comparison, so no honest verdict is
          available. The reasons are in the metric table below.
        </p>
      ) : (
        <>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <span className="text-sm">
              <span className="text-muted-foreground">{metric}</span>{" "}
              <ReadoutValue value={verdict.model} className="text-xl font-semibold" />
            </span>
            <span className="text-sm">
              <span className="text-muted-foreground">baseline</span>{" "}
              <ReadoutValue value={verdict.baseline} className="text-xl font-semibold" />
            </span>
            {verdict.delta != null && (
              <span className="text-sm text-muted-foreground">
                {verdict.delta > 0 ? "+" : ""}
                {verdict.delta.toFixed(3)}
              </span>
            )}
          </div>
          {verdict.ci && (
            <p className="mt-1 text-xs text-muted-foreground">
              95% interval for this {metric}: [<ReadoutValue value={verdict.ci[0]} />,{" "}
              <ReadoutValue value={verdict.ci[1]} />] (bootstrap over the test set, unpaired)
            </p>
          )}
          {/* Within noise by the interval: the only such verdict with no noise floor. */}
          {verdict.kind === "within-noise" && verdict.noiseFloor == null && (
            <p className="mt-2 text-sm">
              The baseline's number sits inside that interval, so this test set cannot tell the two
              models apart.
            </p>
          )}
          {verdict.kind === "within-noise" && verdict.noiseFloor != null && (
            <p className="mt-2 text-sm">
              The margin is <ReadoutValue value={Math.abs(verdict.delta ?? 0)} precision={3} />, and
              repeat measurements of the same compound in this dataset disagree by{" "}
              <ReadoutValue value={verdict.noiseFloor} unit={scorecard.unit} precision={3} />. You
              cannot tell these two models apart with this data — treat them as equivalent and
              prefer the simpler one.
            </p>
          )}
          <p className="mt-2 text-sm text-muted-foreground">
            The baseline is {describeBaseline(scorecard)} on the same dataset and the same split.
            {/* This claim is specifically about fingerprint baselines -- it is
                false about e.g. a chemprop baseline, and this page's whole
                purpose is to tell a scientist the truth about their model. */}
            {scorecard.baseline_engine_id.startsWith("ecfp4-") && (
              <>
                {" "}
                In published benchmarks a fingerprint baseline places mid-field against
                purpose-built models — a model that cannot beat one has not earned its complexity.
              </>
            )}
          </p>
        </>
      )}

      <HonestyStats scorecard={scorecard} />
    </div>
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
            Tune conditions against the <span className="font-medium">validation</span> column. The
            test column is the verdict: every time you retrain and read it, it becomes a little less
            of a held-out set, and the number it reports drifts upward for reasons that have nothing
            to do with the model.
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            This run has no validation score — its split declared no validation partition, or it was
            trained before validation was measured. There is nothing here to tune against except the
            test column, which is the situation to avoid.
          </p>
        )}
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Metric</th>
              {validation && (
                <th className="pb-2 pr-4 font-medium">
                  Validation
                  <span className="ml-1 normal-case text-[10px]">tune here</span>
                </th>
              )}
              <th className="pb-2 pr-4 font-medium">
                Test
                <span className="ml-1 normal-case text-[10px]">the verdict</span>
              </th>
              <th className="pb-2 font-medium">{scorecard.baseline_is_self ? "" : "Baseline"}</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const value = metrics[name];
              const reason = undefinedReasons[name];
              const isPrimary = name === scorecard.primary_metric;
              return (
                <tr key={name} className="border-b last:border-0 align-top">
                  <td className={cn("py-2 pr-4", isPrimary && "font-medium")}>
                    {metricLabel(name)}
                    {isPrimary && (
                      <span className="ml-2 text-xs font-normal text-muted-foreground">
                        primary
                      </span>
                    )}
                  </td>
                  {validation && (
                    <td className="py-2 pr-4">
                      <ReadoutValue value={validation[name]} />
                    </td>
                  )}
                  <td className="py-2 pr-4">
                    {value == null && reason ? (
                      // Never a bare blank where a number belongs. The reason is
                      // written for a scientist and says what to do about it.
                      <span className="text-xs text-warning">{reason}</span>
                    ) : (
                      <ReadoutValue value={value} />
                    )}
                  </td>
                  <td className="py-2">
                    {scorecard.baseline_is_self ? (
                      <span className="text-xs text-muted-foreground">is the baseline</span>
                    ) : (
                      <ReadoutValue value={baseline[name]} className="text-muted-foreground" />
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

function WorstRows({ scorecard }: { scorecard: ScorecardResponse }) {
  if (scorecard.worst_rows.length === 0) return null;

  // Grouped by Murcko scaffold, so the answer reads "it fails on the
  // sulfonamides" rather than as twenty unrelated misses.
  const byScaffold = new Map<string, typeof scorecard.worst_rows>();
  for (const row of scorecard.worst_rows) {
    const existing = byScaffold.get(row.scaffold);
    if (existing) existing.push(row);
    else byScaffold.set(row.scaffold, [row]);
  }
  const groups = [...byScaffold.entries()].sort((a, b) => b[1].length - a[1].length);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Where it fails</CardTitle>
        <p className="text-sm text-muted-foreground">
          The {scorecard.worst_rows.length} worst predictions in the test set, grouped by Murcko
          scaffold. A cluster here is worth more than any aggregate score: it tells you which
          chemistry the model has not learned.
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {groups.map(([scaffold, rows]) => (
          <div key={scaffold}>
            <div className="mb-2 flex items-baseline gap-2">
              <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                {rows.length} compound{rows.length === 1 ? "" : "s"}
              </span>
              <span className="truncate font-mono text-xs text-muted-foreground">
                {scaffold || "no ring system"}
              </span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {rows.map((row) => (
                <div
                  key={row.structure}
                  className="flex h-full flex-col items-center gap-2 rounded-lg border border-border p-3"
                >
                  <StructureThumbnail smiles={row.structure} size={110} />
                  <dl className="w-full space-y-0.5 text-xs">
                    <div className="flex justify-between gap-2">
                      <dt className="text-muted-foreground">measured</dt>
                      <dd>
                        <ReadoutValue value={row.actual} unit={scorecard.unit} precision={2} />
                      </dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-muted-foreground">predicted</dt>
                      <dd>
                        <ReadoutValue value={row.predicted} unit={scorecard.unit} precision={2} />
                      </dd>
                    </div>
                    <div className="flex justify-between gap-2 border-t pt-0.5">
                      <dt className="text-muted-foreground">off by</dt>
                      <dd className="font-medium text-warning">
                        <ReadoutValue value={Math.abs(row.residual)} precision={2} />
                      </dd>
                    </div>
                    {/* `WorstRow.similarity` exists, per its own docstring, "so
                        the triage grid can flag individual out-of-distribution
                        compounds" -- and nothing read it. A bad prediction on a
                        compound unlike anything trained on is a different
                        finding from a bad prediction on a familiar one. */}
                    {row.similarity != null && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-muted-foreground">nearest train</dt>
                        <dd
                          className={
                            row.similarity < 0.3 ? "font-medium text-warning" : "tabular-nums"
                          }
                        >
                          {row.similarity.toFixed(2)}
                          {row.similarity < 0.3 && (
                            <span className="ml-1 text-[10px] uppercase">out of domain</span>
                          )}
                        </dd>
                      </div>
                    )}
                  </dl>
                </div>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ScorecardView({ scorecard }: { scorecard: ScorecardResponse }) {
  return (
    <div className="space-y-4">
      <VerdictBand scorecard={scorecard} />
      {/* The parity plot sits directly under the verdict because it is the one
          view that can contradict it: a model can beat its baseline and still
          be predicting the dataset mean, and only the scatter shows that. */}
      <ScorecardDiagnostics scorecard={scorecard} />
      {/* Before the metric table: the ESOL run's most actionable finding was
          that 8 of its 20 worst predictions had no ring system at all. An
          aggregate cannot say that, and a table of aggregates should not
          outrank it. */}
      <WorstRows scorecard={scorecard} />
      <SplitComparison scorecard={scorecard} />
      <MetricTable scorecard={scorecard} />
      <Conditions scorecard={scorecard} />
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
  if (names.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">How it was trained</CardTitle>
        <p className="text-sm text-muted-foreground">
          The resolved settings behind these numbers. Reproducing this Protocol means this engine,
          these conditions, and the Dataset it cites.
        </p>
      </CardHeader>
      <CardContent>
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
                  {scorecard.baseline_is_self ? "—" : format(baseline[name])}
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
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
