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
import { splitTitle } from "@/shared/lib/split";
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
export function ScorecardDiagnostics({
  scorecard,
  mode,
}: { scorecard: ScorecardResponse; mode?: "overview" | "detail" }) {
  const isClassification = scorecard.prediction_kind === "probability";

  return (
    <>
      {mode !== "detail" && (
        <ParitySection scorecard={scorecard} isClassification={isClassification} />
      )}
      {mode !== "overview" && <ScaffoldErrorSection scorecard={scorecard} />}
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
            ? "How the model scores active and inactive compounds. Less overlap means it is easier to tell the two apart."
            : "Each point is a test compound. The closer it is to the diagonal line, the closer the prediction is to the measured value."}
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
            caption="Each group is shown as a share of its own class, so the shapes can be compared even when active compounds are rare."
          />
        ) : (
          <ParityChart
            points={scorecard.parity}
            unit={scorecard.unit}
            noiseFloor={scorecard.noise_floor}
            caption={
              scorecard.noise_floor != null
                ? "The dashed line is a perfect prediction. The shaded band shows the average spread of repeated measurements as context for the errors."
                : scorecard.parity.some((point) => point.similarity != null)
                  ? "The dashed line is a perfect prediction. Points are shaded by how similar the compound is to the training set."
                  : // No similarity to shade by -- a sequence dataset has no Tanimoto
                    // neighbour. `ParityChart` already drops the shading and its legend
                    // in that case; promising it in the caption was the last place the
                    // card still claimed a measurement it never made.
                    "The dashed line is a perfect prediction."
            }
          />
        )}

        {scorecard.residual_histogram && scorecard.residual_histogram.counts.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              How far off are the predictions?
            </p>
            <HistogramChart
              bins={scorecard.residual_histogram}
              xLabel={`predicted − measured${scorecard.unit ? ` (${scorecard.unit})` : ""}`}
              height={180}
              reference={{ at: 0, label: "no error" }}
              caption="Zero is a perfect prediction. Values to the right are predictions that were too high; values to the left were too low."
            />
          </div>
        )}

        {scorecard.calibration.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Do the probability estimates match reality?
            </p>
            <BinnedCurveChart
              bins={scorecard.calibration}
              xLabel="predicted probability"
              yLabel="observed active rate"
              diagonal
              height={200}
              caption="Near the diagonal, predicted chances match observed results: among compounds given an 80% chance, about 80% should be active. Bigger dots contain more compounds. Small groups give less certain estimates."
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Which chemistry the model has not learned, systematically rather than anecdotally. */
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
          See individual errors in the Test compounds tab.
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

/** One metric's spread across the draws, or why it has none. */
function DrawSpread({
  row,
  completed,
}: {
  row: { mean: number | null; sd: number | null; n: number } | undefined;
  completed: number;
}) {
  // A metric the server did not summarize at all. Not measured, and not claimed.
  if (!row) return <span className="text-xs text-muted-foreground">Not measured</span>;
  if (row.n === 0) {
    return <span className="text-xs text-warning">Undefined in every draw</span>;
  }
  if (row.sd == null) {
    // One usable draw is a number with nothing under it. Printing "± 0" would claim a
    // stability that was never measured.
    return (
      <span>
        <ReadoutValue value={row.mean} className="text-muted-foreground" />
        <span className="text-xs text-muted-foreground">
          {" from one draw, so there is no spread to report"}
        </span>
      </span>
    );
  }
  return (
    <span>
      <ReadoutValue value={row.mean} className="text-muted-foreground" />
      <span className="text-xs text-muted-foreground">
        {/* "3 of 5 draws" rather than a second clause: a count below the number of
            completed draws means this metric was undefined in some of them, which is a
            different story from a draw that never finished. The warning under the table
            tells that one, and merging the two would blame the wrong thing. */}
        {row.n < completed
          ? ` ± ${row.sd.toFixed(3)} across ${row.n} of ${completed} draws`
          : ` ± ${row.sd.toFixed(3)} across ${row.n} draws`}
      </span>
    </span>
  );
}

/**
 * How much the score moves when the split is drawn again.
 *
 * The scored number came from one test set, and which compounds landed in it was
 * a draw. Training again on other draws of the same kind of split is the only
 * thing that says whether a score is a property of the model or of that choice.
 *
 * Renders nothing only when no draw was asked for. Every other absence has a
 * reason and prints it -- including a partial one: a spread over two draws when
 * five were requested is not the measurement that was asked for, so the table
 * and the warning appear together rather than the table alone.
 */
export function SplitDrawSpread({ scorecard }: { scorecard: ScorecardResponse }) {
  const summary = scorecard.replicate_summary;
  const unavailable = scorecard.replicate_unavailable;
  if (!summary && !unavailable) return null;

  const metrics = (scorecard.metrics ?? {}) as Record<string, number | null>;
  const completed = scorecard.replicate_seeds?.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Score across split draws</CardTitle>
        <p className="text-sm text-muted-foreground">
          {completed > 0
            ? `The same engine and settings, trained again on ${completed === 1 ? "one more draw" : `${completed} more draws`} of the split. Each draw divides the same compounds into training and test sets again. A lead over the baseline smaller than this spread is not evidence of a better model.`
            : "Training on further draws of the split shows how much the score depends on which compounds landed in the test set."}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {summary && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="pb-2 pr-4 font-medium">Metric</th>
                <th className="pb-2 pr-4 font-medium">
                  {splitTitle(scorecard.split_strategy)} split (scored)
                </th>
                <th className="pb-2 font-medium">Across draws</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(metrics).map((name) => (
                <tr key={name} className="border-b align-top last:border-0">
                  <td className="py-2 pr-4">{metricLabel(name)}</td>
                  <td className="py-2 pr-4">
                    <ReadoutValue value={metrics[name]} />
                  </td>
                  <td className="py-2">
                    <DrawSpread row={summary[name]} completed={completed} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {/* Rendered whether or not the table is, and never instead of it. */}
        {unavailable && <p className="text-xs text-warning">{unavailable}</p>}
      </CardContent>
    </Card>
  );
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
        <CardTitle className="text-base">
          {splitTitle(scorecard.split_strategy)} split versus random split
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          The same engine and settings, trained and scored on a random split of the same data.
          Consistent differences across metrics indicate split-induced optimism.
        </p>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Metric</th>
              <th className="pb-2 pr-4 font-medium">
                {splitTitle(scorecard.split_strategy)} split (scored)
              </th>
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
