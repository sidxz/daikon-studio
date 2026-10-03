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
          {isClassification ? "Predicted probability by true class" : "Predicted against measured"}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          {isClassification
            ? "Predicted probability of every test compound, split by true class. Clear separation means actives are ranked above inactives; overlap marks where any threshold will misclassify."
            : "All test compounds. Points near the diagonal are accurate predictions; a cloud flattened toward the middle means predictions regress to the dataset mean."}
          {scorecard.parity_sampled_from
            ? ` Showing ${scorecard.parity.length.toLocaleString()} of ${scorecard.parity_sampled_from.toLocaleString()} test compounds.`
            : ""}
        </p>
      </CardHeader>
      <CardContent className="space-y-5">
        {isClassification ? (
          <SplitHistogramChart
            bins={probabilityByClass(scorecard.parity)}
            xLabel="Predicted P(active)"
            series={["Active", "Inactive"]}
            yLabel="Fraction of class"
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
              caption="On the diagonal, probabilities are calibrated: of compounds predicted at 0.8, 80% are active. Off it, outputs are scores rather than probabilities, though ranking may still be correct. Dot size shows compounds per bin."
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
          MAE of test compounds, binned by NN similarity (equal-count bins). Shows whether the
          applicability-domain coverage above tracks error.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <BinnedCurveChart
          bins={bins}
          xLabel="Tanimoto similarity to nearest training compound"
          yLabel={`mean absolute error${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
          caption="Bins hold equal numbers of compounds, so no bin rests on fewer data than another."
        />
        <p className="text-sm">
          {rises ? (
            <>
              MAE is{" "}
              <span className="font-medium">
                {ratio ? `${ratio.toFixed(1)}× higher` : "higher"}
              </span>{" "}
              in the lowest-similarity bin (
              <ReadoutValue value={first.value} precision={3} />) than in the highest (
              <ReadoutValue value={last.value} precision={3} />
              ). Predictions on compounds dissimilar to the training set are less reliable for this
              model.
            </>
          ) : (
            <>
              MAE is <span className="font-medium">not</span> higher in the lowest-similarity bin
              than in the highest. Either the model generalizes beyond its training chemistry, or
              Tanimoto similarity does not capture what makes these compounds difficult. Either way,
              applicability-domain coverage is a weak guide to error here.
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
          Median absolute error per Bemis–Murcko scaffold across the full test set, highest first.
          The individual largest errors are listed above.
        </p>
      </CardHeader>
      <CardContent>
        <HorizontalBarChart
          data={data}
          xLabel={`median absolute error${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
          height={Math.max(160, data.length * 26 + 50)}
          format={(value) => value.toFixed(2)}
          caption="Only scaffolds with at least three test compounds are shown."
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
        <CardTitle className="text-base">Scaffold split versus random split</CardTitle>
        <p className="text-sm text-muted-foreground">
          The same engine and settings, trained and scored on a random split of the same compounds.
          Consistent differences across metrics indicate split-induced optimism.
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
