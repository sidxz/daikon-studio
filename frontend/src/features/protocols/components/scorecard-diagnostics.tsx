"use client";

import {
  BinnedCurveChart,
  HistogramChart,
  HorizontalBarChart,
  ParityChart,
  type SplitBins,
  SplitHistogramChart,
} from "@/shared/components/charts";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { metricLabel } from "../types";

/**
 * The part of the Scorecard that says *where* the model is wrong.
 *
 * The verdict band above answers "how good is this number". Everything here
 * answers the question a scientist actually acts on, which the aggregates
 * cannot: on which compounds, on which chemistry, and at what distance from the
 * training set does this model stop working.
 *
 * Nothing here is newly measured. Every chart is derived from the same
 * `actual`/`predicted`/`structures` the headline metrics came from -- data that
 * was already in the scorecard blob and was being discarded before it reached
 * the page.
 */
export function ScorecardDiagnostics({ scorecard }: { scorecard: ScorecardResponse }) {
  const isClassification = scorecard.prediction_kind === "probability";

  return (
    <>
      <ParitySection scorecard={scorecard} isClassification={isClassification} />
      <ApplicabilitySection scorecard={scorecard} />
      <ScaffoldErrorSection scorecard={scorecard} />
    </>
  );
}

function ParitySection({
  scorecard,
  isClassification,
}: {
  scorecard: ScorecardResponse;
  isClassification: boolean;
}) {
  if (scorecard.parity.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {isClassification ? "Predicted probability against truth" : "Predicted against measured"}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          {isClassification
            ? "Every test compound, its true class against the probability the model gave it. Well-separated clouds mean the model can rank; overlap in the middle is where the threshold decision actually costs something."
            : "Every test compound in the test set, not just the twenty worst. A cloud that hugs the diagonal is a working model; a cloud that flattens toward the middle is a model predicting the dataset average and scoring respectably for it."}
          {scorecard.parity_sampled_from
            ? ` Showing ${scorecard.parity.length.toLocaleString()} of ${scorecard.parity_sampled_from.toLocaleString()} test compounds.`
            : ""}
        </p>
      </CardHeader>
      <CardContent className="space-y-5">
        {isClassification ? (
          <SplitHistogramChart
            bins={probabilityByClass(scorecard.parity)}
            xLabel="predicted probability of being active"
            series={["truly active", "truly inactive"]}
            yLabel="share of class"
            height={220}
            caption="Each class normalised to its own size, so the shapes stay comparable on an unbalanced test set. Two humps pushed to opposite ends is a model that separates the classes; overlap in the middle is the region where whatever threshold you pick will be wrong about something."
          />
        ) : (
          <ParityChart
            points={scorecard.parity}
            unit={scorecard.unit}
            noiseFloor={scorecard.noise_floor}
            caption={
              scorecard.noise_floor != null
                ? "The dashed line is a perfect prediction; the shaded band is the assay noise floor. A point inside the band is as accurate as this data can prove anything is."
                : "The dashed line is a perfect prediction. Points are shaded by how similar the compound is to the training set."
            }
          />
        )}

        {scorecard.residual_histogram && scorecard.residual_histogram.counts.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Residuals
            </p>
            <HistogramChart
              bins={scorecard.residual_histogram}
              xLabel={`predicted − measured${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
              height={180}
              reference={{ at: 0, label: "no error" }}
              caption="Centred on zero means the model is wrong in both directions equally. Centred to one side is bias — a model that is systematically optimistic or pessimistic, which no error metric on this page reports."
            />
          </div>
        )}

        {scorecard.calibration.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Calibration
            </p>
            <BinnedCurveChart
              bins={scorecard.calibration}
              xLabel="predicted probability"
              yLabel="observed active rate"
              diagonal
              height={200}
              caption="On the diagonal, a predicted 0.8 means 80% of those compounds really were active — the probability can be read as one. Off it, the model may still rank compounds correctly, but its numbers are scores rather than probabilities. Dot size is how many compounds fell in each band."
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * The applicability claim, checked.
 *
 * The verdict band states a single coverage percentage. This is the evidence for
 * or against it: error should rise as compounds get less like the training set,
 * and if it does not, the applicability domain is not buying this model anything.
 */
function ApplicabilitySection({ scorecard }: { scorecard: ScorecardResponse }) {
  const bins = scorecard.error_by_similarity;
  if (bins.length === 0) return null;

  const first = bins[0];
  const last = bins[bins.length - 1];
  const rises = first.value > last.value;
  const ratio = last.value === 0 ? null : first.value / last.value;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Does distance from training predict error?</CardTitle>
        <p className="text-sm text-muted-foreground">
          Mean absolute error against how similar each test compound is to its nearest training
          compound, in equal-sized groups. This is the applicability number on the verdict band,
          shown as evidence rather than asserted.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <BinnedCurveChart
          bins={bins}
          xLabel="nearest-neighbour Tanimoto to the training set"
          yLabel={`mean absolute error${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
          caption="Dot size is how many compounds are in each group; every group holds the same number, so a wobble at one end is not a small-sample artefact."
        />
        <p className="text-sm">
          {rises ? (
            <>
              Error is{" "}
              <span className="font-medium">
                {ratio ? `${ratio.toFixed(1)}×` : "measurably"} higher
              </span>{" "}
              on the least familiar compounds (
              <ReadoutValue value={first.value} precision={3} />) than on the most familiar (
              <ReadoutValue value={last.value} precision={3} />
              ). The applicability domain is real for this model — treat predictions on unfamiliar
              chemistry as less trustworthy, in that proportion.
            </>
          ) : (
            <>
              Error does <span className="font-medium">not</span> rise as compounds get less like
              the training set. Either the model generalizes past its training chemistry, or the
              similarity measure is not capturing what makes a compound hard here — in both cases
              the applicability percentage above is not the caveat it appears to be.
            </>
          )}
        </p>
      </CardContent>
    </Card>
  );
}

/** Which chemistry the model has not learned — systematically, not anecdotally. */
function ScaffoldErrorSection({ scorecard }: { scorecard: ScorecardResponse }) {
  const families = scorecard.scaffold_errors;
  if (families.length === 0) return null;

  const data = families.map((family) => ({
    label: family.scaffold ? truncate(family.scaffold) : "no ring system",
    value: family.median_error,
    detail: `${family.count} compound${family.count === 1 ? "" : "s"}`,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Error by scaffold family</CardTitle>
        <p className="text-sm text-muted-foreground">
          Median absolute error per Murcko scaffold across the whole test set, worst first. The
          worst-predictions grid above shows twenty individual misses; this says which families they
          come from, which is the version you can act on.
        </p>
      </CardHeader>
      <CardContent>
        <HorizontalBarChart
          data={data}
          xLabel={`median absolute error${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
          height={Math.max(160, data.length * 26 + 50)}
          format={(value) => value.toFixed(2)}
          caption="Only families with at least three test compounds appear — a median over one or two is not a median."
        />
      </CardContent>
    </Card>
  );
}

/**
 * Predicted probabilities, split by the class each compound actually belongs to.
 *
 * A parity scatter is the wrong shape for classification: `actual` is only ever
 * 0 or 1, so every point lands in one of two vertical strips and the y = x line
 * -- the entire read of a parity plot -- passes through a region no compound can
 * occupy. Two overlaid distributions answer the question that plot was standing
 * in for, which is whether the model separates the classes at all.
 */
function probabilityByClass(points: { actual: number; predicted: number }[], bins = 20): SplitBins {
  const edges = Array.from({ length: bins + 1 }, (_, index) => index / bins);
  const active = new Array<number>(bins).fill(0);
  const inactive = new Array<number>(bins).fill(0);
  for (const point of points) {
    // The last bin closes at 1.0 inclusive, so a confident P = 1.0 prediction is
    // counted rather than silently dropped off the end.
    const index = Math.min(Math.floor(point.predicted * bins), bins - 1);
    if (point.actual > 0.5) active[index] += 1;
    else inactive[index] += 1;
  }
  // `train`/`test` are this type's field names, not a claim about partitions --
  // the `series` labels are what the reader sees.
  return { edges, train: active, test: inactive };
}

function truncate(smiles: string): string {
  return smiles.length > 22 ? `${smiles.slice(0, 21)}…` : smiles;
}

/**
 * How the model scored under the split it was actually judged on versus the
 * easier one.
 *
 * The verdict band reads one metric out of `random_split_metrics` and discards
 * the rest of the dict, which is already on the wire. Every metric tells the
 * same story with a different sensitivity, and seeing them agree (or not) is
 * what makes the optimism gap credible.
 */
export function SplitComparison({ scorecard }: { scorecard: ScorecardResponse }) {
  const random = scorecard.random_split_metrics as Record<string, number | null> | null;
  const metrics = (scorecard.metrics ?? {}) as Record<string, number | null>;
  const undefinedReasons = (scorecard.random_split_metrics_undefined ?? {}) as Record<
    string,
    string
  >;
  if (!random) return null;

  const names = Object.keys(metrics);
  if (names.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What an easier split would have said</CardTitle>
        <p className="text-sm text-muted-foreground">
          The same model, scored on a random re-split of the same rows. Every metric, not just the
          primary one — the gap is only convincing if the metrics agree about it.
        </p>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Metric</th>
              <th className="pb-2 pr-4 font-medium">Scaffold split (scored)</th>
              <th className="pb-2 font-medium">Random split</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const reason = undefinedReasons[name];
              return (
                <tr key={name} className="border-b align-top last:border-0">
                  <td className="py-2 pr-4">{metricLabel(name)}</td>
                  <td className="py-2 pr-4">
                    <ReadoutValue value={metrics[name]} />
                  </td>
                  <td className="py-2">
                    {random[name] == null && reason ? (
                      // The random split's own reason, never the scored split's:
                      // a class that survives one partition can vanish in the other.
                      <span className="text-xs text-warning">{reason}</span>
                    ) : (
                      <ReadoutValue value={random[name]} className="text-muted-foreground" />
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
