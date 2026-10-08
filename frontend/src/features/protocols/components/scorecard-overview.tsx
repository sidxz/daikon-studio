"use client";

import { ReadoutValue } from "@/shared/components/readout-value";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Progress } from "@/shared/components/ui/progress";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { ArrowDownUp, Crosshair, Lightbulb, Target, TrendingUp } from "lucide-react";
import { type ReactNode, useId, useState } from "react";
import { useScorecardTolerance } from "../hooks/use-protocols";

function SummaryCard({
  label,
  detail,
  icon,
  children,
  accent = false,
}: {
  label: string;
  detail: string;
  icon: ReactNode;
  children: ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "min-w-0 rounded-lg border bg-card p-5",
        accent && "border-primary/25 bg-primary/[0.03]",
      )}
    >
      <div className="mb-4 flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">{label}</h3>
        <span className="text-muted-foreground" aria-hidden="true">
          {icon}
        </span>
      </div>
      {children}
      <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{detail}</p>
    </div>
  );
}

function Percentage({ value }: { value: number | null | undefined }) {
  return value == null ? (
    <span className="text-muted-foreground">N/A</span>
  ) : (
    <>
      {(value * 100).toFixed(1)}
      <span className="ml-0.5 text-xl text-muted-foreground">%</span>
    </>
  );
}

function ToleranceCard({
  protocolId,
  scorecard,
  onToleranceChange,
}: {
  protocolId: string;
  scorecard: ScorecardResponse;
  onToleranceChange?: (value: number | null) => void;
}) {
  const inputId = useId();
  const [input, setInput] = useState("");
  const [tolerance, setTolerance] = useState<number | null>(null);
  const result = useScorecardTolerance(protocolId, scorecard.target, tolerance);
  const numeric = input.trim() === "" ? null : Number(input);
  const valid = numeric !== null && Number.isFinite(numeric) && numeric >= 0;
  const changed = numeric !== tolerance;
  // A result always belongs to the submitted value. Never present the old
  // percentage beside a newly edited tolerance or a still-pending request.
  const data = !changed ? result.data : undefined;
  const fraction = data && data.test_count > 0 ? data.within_count / data.test_count : null;
  return (
    <SummaryCard
      label="Within acceptable error"
      detail="Choose the largest difference from a measured value that would still be useful to you."
      icon={<Crosshair className="size-4" />}
    >
      <div className="text-3xl font-semibold tracking-tight" aria-live="polite">
        {result.isFetching && !changed ? (
          <span className="text-sm font-normal text-muted-foreground">Checking…</span>
        ) : fraction == null ? (
          <span className="text-xl font-medium text-muted-foreground">Set your tolerance</span>
        ) : (
          <Percentage value={fraction} />
        )}
      </div>
      <form
        className="mt-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (valid) {
            setTolerance(numeric);
            onToleranceChange?.(numeric);
          }
        }}
      >
        <label htmlFor={inputId} className="mb-1.5 block text-xs text-muted-foreground">
          Acceptable error{scorecard.unit ? ` (${scorecard.unit})` : " (target units)"}
        </label>
        <div className="flex gap-2">
          <Input
            id={inputId}
            type="number"
            min="0"
            step="any"
            value={input}
            placeholder="Enter a value"
            className="h-8 min-w-0"
            onChange={(event) => {
              const next = event.target.value;
              setInput(next);
              // Editing back to the checked value shows its result again.
              onToleranceChange?.(
                next.trim() !== "" && Number(next) === tolerance ? tolerance : null,
              );
            }}
          />
          <Button
            type="submit"
            variant="outline"
            size="sm"
            disabled={!valid || (result.isFetching && !changed)}
          >
            Check
          </Button>
        </div>
      </form>
      {fraction != null && data && (
        <div className="mt-3">
          <Progress value={fraction * 100} className="h-1.5" />
          <p className="mt-2 text-xs text-muted-foreground">
            {data.within_count.toLocaleString()} of {data.test_count.toLocaleString()} test
            predictions within ±<ReadoutValue value={data.tolerance} unit={scorecard.unit} />.
          </p>
        </div>
      )}
      {result.isError && !changed && (
        <p role="alert" className="mt-2 text-xs text-destructive">
          Could not check this tolerance.{" "}
          <button type="button" className="underline" onClick={() => result.refetch()}>
            Try again
          </button>
        </p>
      )}
      {data?.test_count === 0 && (
        <p className="mt-2 text-xs text-muted-foreground">No test predictions to evaluate.</p>
      )}
    </SummaryCard>
  );
}

function ErrorInterpretation({ scorecard }: { scorecard: ScorecardResponse }) {
  const summary = scorecard.regression_summary;
  const mae = scorecard.metrics.mae;
  const baseline = scorecard.baseline_metrics?.mae ?? null;
  const improvement =
    !scorecard.baseline_is_self && mae != null && baseline != null && baseline > 0
      ? (baseline - mae) / baseline
      : null;
  const r2 = scorecard.metrics.r2;
  return (
    <div className="rounded-lg border bg-card px-5 py-4">
      <div className="flex items-center gap-2">
        <Lightbulb className="size-4 text-primary" aria-hidden="true" />
        <h3 className="text-sm font-medium">What this means</h3>
      </div>
      <div className="mt-3 grid gap-4 text-sm sm:grid-cols-2">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Larger errors</p>
          {summary ? (
            <p>
              At least 9 in 10 test predictions were within{" "}
              <span className="font-medium">
                <ReadoutValue value={summary.absolute_error_p90} unit={scorecard.unit} />
              </span>{" "}
              of the measured value.
            </p>
          ) : (
            <p className="text-muted-foreground">Error range is not available.</p>
          )}
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Tendency to predict high or low</p>
          {summary ? (
            <p>
              {summary.mean_signed_error === 0 ? (
                "Overestimates and underestimates balanced out on average."
              ) : (
                <>
                  Predictions were{" "}
                  <span className="font-medium">
                    <ReadoutValue
                      value={Math.abs(summary.mean_signed_error)}
                      unit={scorecard.unit}
                    />{" "}
                    too {summary.mean_signed_error > 0 ? "high" : "low"}
                  </span>{" "}
                  on average.
                </>
              )}
            </p>
          ) : (
            <p className="text-muted-foreground">Prediction bias is not available.</p>
          )}
        </div>
      </div>
      {improvement != null && (
        <p className="mt-4 border-t pt-3 text-sm">
          {improvement === 0 ? (
            "Average error matches the comparison model."
          ) : (
            <>
              Average error is{" "}
              <span className="font-medium">
                {(Math.abs(improvement) * 100).toFixed(1)}% {improvement > 0 ? "lower" : "higher"}
              </span>{" "}
              than the comparison model (<ReadoutValue value={baseline} unit={scorecard.unit} />
              ). This is the observed difference on this test set.
            </>
          )}
        </p>
      )}
      {r2 != null && r2 < 0 && (
        <p className="mt-3 text-sm text-warning">
          The negative R² means that always predicting the test-set average would have produced less
          squared error.
        </p>
      )}
    </div>
  );
}

export function ScorecardOverview({
  scorecard,
  protocolId,
  onToleranceChange,
}: {
  scorecard: ScorecardResponse;
  protocolId?: string;
  onToleranceChange?: (value: number | null) => void;
}) {
  const binary = scorecard.prediction_kind === "probability";
  const summary = scorecard.classification_summary;
  const count = scorecard.test_count ?? scorecard.parity_sampled_from ?? scorecard.parity?.length;
  return (
    <section className="space-y-4" aria-label="Performance overview">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">
            How well does your model predict?
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {count != null ? (
              <>
                Tested on{" "}
                <span className="font-medium text-foreground">
                  {count.toLocaleString()} compounds
                </span>{" "}
                the model did not train on.
              </>
            ) : (
              "Results on compounds the model did not train on."
            )}
            {summary && (
              <>
                {" "}
                {(summary.true_positive + summary.false_negative).toLocaleString()} active ·{" "}
                {(summary.true_negative + summary.false_positive).toLocaleString()} inactive.
              </>
            )}
          </p>
        </div>
        <Badge variant="outline" className="font-normal">
          {binary ? "Binary prediction" : "Continuous prediction"}
        </Badge>
      </div>
      <div className="grid items-stretch gap-3 md:grid-cols-3">
        {binary ? (
          <>
            <SummaryCard
              label="Precision"
              detail={
                !summary
                  ? "Precision is not available for this result."
                  : summary.precision == null
                    ? "No compounds were predicted active, so precision cannot be measured."
                    : "Of the compounds predicted active, this share actually was active."
              }
              icon={<Target className="size-4" />}
              accent
            >
              <div className="text-3xl font-semibold tracking-tight">
                <Percentage value={summary?.precision} />
              </div>
              {summary && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {summary.true_positive.toLocaleString()} confirmed out of{" "}
                  {(summary.true_positive + summary.false_positive).toLocaleString()} predicted
                  active
                </p>
              )}
            </SummaryCard>
            <SummaryCard
              label="Recall"
              detail={
                !summary
                  ? "Recall is not available for this result."
                  : summary.recall == null
                    ? "No active compounds were in the test set, so recall cannot be measured."
                    : "Of all the truly active compounds, this share was found by the model."
              }
              icon={<Crosshair className="size-4" />}
            >
              <div className="text-3xl font-semibold tracking-tight">
                <Percentage value={summary?.recall} />
              </div>
              {summary && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {summary.true_positive.toLocaleString()} found out of{" "}
                  {(summary.true_positive + summary.false_negative).toLocaleString()} active
                </p>
              )}
            </SummaryCard>
            <SummaryCard
              label="False alarms"
              detail="Compounds predicted active that were actually inactive. These could lead to unnecessary follow-up experiments."
              icon={<ArrowDownUp className="size-4" />}
            >
              <div className="text-3xl font-semibold tracking-tight">
                {summary?.false_positive.toLocaleString() ?? "N/A"}
              </div>
              {summary && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {summary.false_negative.toLocaleString()} active compounds were also missed
                </p>
              )}
            </SummaryCard>
          </>
        ) : (
          <>
            <SummaryCard
              label="Average prediction error"
              detail="The average distance between a prediction and its measured value. Lower is better."
              icon={<Target className="size-4" />}
              accent
            >
              <div className="text-3xl font-semibold tracking-tight">
                <ReadoutValue value={scorecard.metrics.mae} unit={scorecard.unit} />
              </div>
              <p className="mt-2 text-xs text-muted-foreground">Mean absolute error · MAE</p>
            </SummaryCard>
            <SummaryCard
              label="Variation explained · R²"
              detail="1 = perfect predictions. 0 = always guessing the test-set average. Negative = worse than that. This is not a percentage of correct predictions."
              icon={<TrendingUp className="size-4" />}
            >
              <div className="text-3xl font-semibold tracking-tight">
                <ReadoutValue value={scorecard.metrics.r2} />
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                {scorecard.metrics.r2 == null
                  ? (scorecard.metrics_undefined?.r2 ?? "R² could not be measured.")
                  : "Higher is better"}
              </p>
            </SummaryCard>
            {protocolId ? (
              <ToleranceCard
                key={scorecard.target}
                protocolId={protocolId}
                scorecard={scorecard}
                onToleranceChange={onToleranceChange}
              />
            ) : (
              <SummaryCard
                label="Within acceptable error"
                detail="Open this protocol to check predictions against your own error tolerance."
                icon={<Crosshair className="size-4" />}
              >
                <span className="text-3xl text-muted-foreground">N/A</span>
              </SummaryCard>
            )}
          </>
        )}
      </div>
      {!binary && <ErrorInterpretation scorecard={scorecard} />}
    </section>
  );
}
