"use client";

import { ChartLegend, PlotFigure, baseOptions, useChartTheme } from "@/shared/components/charts";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import type { ClassificationBinResponse, ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import * as Plot from "@observablehq/plot";
import { useCallback, useState } from "react";
import { useScorecardTolerance } from "../hooks/use-protocols";
import { formatCutoff } from "../lib/format-cutoff";

type Point = { lower: number; upper: number; value: number | null; detail: string };
type Series = { label: string; points: Point[] };

function SimilarityCurve({
  series,
  label,
  percentage = false,
}: {
  series: Series[];
  label: string;
  percentage?: boolean;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    if (!theme) return {};
    return {
      ...baseOptions(theme),
      ariaLabel: label,
      x: { label: null, domain: [0, 1], grid: true, ticks: 5 },
      y: {
        label: null,
        grid: true,
        zero: true,
        ...(percentage ? { domain: [0, 100], tickFormat: (value: number) => `${value}%` } : {}),
      },
      marks: series.flatMap((item, index) => [
        Plot.line(item.points, {
          x: (point: Point) => (point.lower + point.upper) / 2,
          y: "value",
          stroke: theme.pair[index % 2],
          strokeWidth: 2,
          ...(index === 1 ? { strokeDasharray: "5,4" } : {}),
        }),
        Plot.dot(item.points, {
          x: (point: Point) => (point.lower + point.upper) / 2,
          y: "value",
          fill: theme.pair[index % 2],
          r: 4,
          symbol: index === 1 ? "triangle" : "circle",
          title: (point: Point) =>
            `${item.label}\nSimilarity ${point.lower.toFixed(3)}–${point.upper.toFixed(3)}\n${point.detail}`,
        }),
      ]),
    };
  }, [theme, series, label, percentage]);
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">{label}</p>
      {theme && series.length > 1 && (
        <ChartLegend
          items={series.map((item, index) => ({
            label: `${item.label}${index === 1 ? " · dashed line, triangles" : " · solid line, circles"}`,
            color: theme.pair[index % 2],
          }))}
        />
      )}
      {theme ? (
        <PlotFigure options={options} height={230} />
      ) : (
        <div className="h-56 animate-pulse bg-muted/30" />
      )}
      <div className="flex justify-between gap-4 text-xs text-muted-foreground">
        <span>Less similar</span>
        <span>More similar</span>
      </div>
      <p className="text-center text-xs text-muted-foreground">Similarity to training compounds</p>
    </div>
  );
}

function rate(misses: number, total: number): number | null {
  return total === 0 ? null : (100 * misses) / total;
}

function rateText(misses: number, total: number, missing: string): string {
  const value = rate(misses, total);
  return value == null ? missing : `${value.toFixed(1)}% (${misses} of ${total})`;
}

function BinarySimilarity({
  bins,
  scorecard,
}: {
  bins: ClassificationBinResponse[];
  scorecard: ScorecardResponse;
}) {
  const series: Series[] = [
    {
      label: "Missed actives",
      points: bins.map((bin) => {
        const { false_negative: missed, true_positive: found } = bin.summary;
        return {
          ...bin,
          value: rate(missed, missed + found),
          detail: rateText(missed, missed + found, "No measured actives in this group"),
        };
      }),
    },
    {
      label: "False alarms",
      points: bins.map((bin) => {
        const { false_positive: alarms, true_negative: rejected } = bin.summary;
        return {
          ...bin,
          value: rate(alarms, alarms + rejected),
          detail: rateText(alarms, alarms + rejected, "No measured inactives in this group"),
        };
      }),
    },
  ];
  return (
    <>
      <SimilarityCurve series={series} label="Mistake rate: lower is better" percentage />
      <p className="text-xs text-muted-foreground">
        Missed actives: the share of measured actives predicted inactive. False alarms: the share of
        measured inactives predicted active. Predictions use the saved decision rule: chance of
        active {scorecard.classification_summary?.cutoff_inclusive ? "≥" : ">"}{" "}
        {formatCutoff(scorecard.cutoff ?? 0.5)}. Small class counts make rates less certain; a
        missing class is shown as unavailable, with a gap in its line.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <caption className="sr-only">Mistakes by similarity group</caption>
          <thead className="border-b text-muted-foreground">
            <tr>
              <th scope="col" className="py-2 pr-4 font-medium">
                Similarity range
              </th>
              <th scope="col" className="py-2 pr-4 font-medium">
                Test compounds
              </th>
              <th scope="col" className="py-2 pr-4 font-medium">
                Missed actives
              </th>
              <th scope="col" className="py-2 font-medium">
                False alarms
              </th>
            </tr>
          </thead>
          <tbody>
            {bins.map((bin) => {
              const s = bin.summary;
              return (
                <tr key={`${bin.lower}-${bin.upper}`} className="border-b last:border-0">
                  <th scope="row" className="py-2 pr-4 font-normal tabular-nums">
                    {bin.lower.toFixed(3)}–{bin.upper.toFixed(3)}
                  </th>
                  <td className="py-2 pr-4">{bin.count}</td>
                  <td className="py-2 pr-4">
                    {rateText(
                      s.false_negative,
                      s.false_negative + s.true_positive,
                      "Unavailable (no actives)",
                    )}
                  </td>
                  <td className="py-2">
                    {rateText(
                      s.false_positive,
                      s.false_positive + s.true_negative,
                      "Unavailable (no inactives)",
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ToleranceSimilarity({
  protocolId,
  scorecard,
  tolerance,
}: {
  protocolId: string;
  scorecard: ScorecardResponse;
  tolerance: number;
}) {
  const result = useScorecardTolerance(protocolId, scorecard.target, tolerance);
  if (result.isError)
    return (
      <p role="alert" className="text-sm">
        Could not load the tolerance comparison.{" "}
        <button type="button" className="underline" onClick={() => result.refetch()}>
          Try again
        </button>
      </p>
    );
  if (!result.data)
    return <p className="text-sm text-muted-foreground">Checking similarity groups…</p>;
  const bins = result.data.by_similarity;
  if (!bins?.length)
    return (
      <p className="text-sm text-muted-foreground">
        Similarity information is unavailable for this tolerance comparison.
      </p>
    );
  const first = bins[0];
  const last = bins[bins.length - 1];
  return (
    <>
      <SimilarityCurve
        label="Within acceptable error: higher is better"
        percentage
        series={[
          {
            label: "Within acceptable error",
            points: bins.map((bin) => ({
              ...bin,
              value: rate(bin.within_count, bin.count),
              detail: rateText(bin.within_count, bin.count, "No test compounds"),
            })),
          },
        ]}
      />
      <p className="text-sm">
        Within ±<ReadoutValue value={tolerance} unit={scorecard.unit} /> of the measured value:{" "}
        {rateText(first.within_count, first.count, "Unavailable")} in the least-similar group
        {bins.length > 1
          ? ` and ${rateText(last.within_count, last.count, "Unavailable")} in the most-similar group`
          : " (the only group)"}
        .
      </p>
    </>
  );
}

export function ScorecardSimilarity({
  scorecard,
  protocolId,
  tolerance = null,
}: {
  scorecard: ScorecardResponse;
  protocolId?: string;
  tolerance?: number | null;
}) {
  const [view, setView] = useState<"error" | "tolerance">("error");
  const binary = scorecard.prediction_kind === "probability";
  const bins = scorecard.error_by_similarity ?? [];
  const binaryBins = scorecard.classification_by_similarity ?? [];
  const available = binary ? binaryBins.length > 0 : bins.length > 0;
  const showTolerance = view === "tolerance" && tolerance != null && protocolId != null;
  const first = bins[0];
  const last = bins[bins.length - 1];
  const label = `Average prediction error${scorecard.unit ? ` (${scorecard.unit})` : ""}: lower is better`;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Are predictions better for familiar compounds?</CardTitle>
        <p className="text-sm text-muted-foreground">
          Test compounds are grouped by how similar they are to their nearest training compound. The
          chart uses all test results. Groups aim for similar sizes; identical similarity scores
          stay together.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {!available ? (
          <p className="text-sm text-muted-foreground">
            There are not enough test compounds with training-similarity data to show a comparison.
          </p>
        ) : binary ? (
          <BinarySimilarity bins={binaryBins} scorecard={scorecard} />
        ) : (
          <>
            <fieldset className="flex flex-wrap gap-2">
              <legend className="sr-only">Similarity chart measure</legend>
              {(
                [
                  ["error", "Average error"],
                  ["tolerance", "Within acceptable error"],
                ] as const
              ).map(([value, text]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={value === (showTolerance ? "tolerance" : "error")}
                  disabled={value === "tolerance" && (tolerance == null || !protocolId)}
                  onClick={() => setView(value)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-xs disabled:opacity-50",
                    value === (showTolerance ? "tolerance" : "error") && "bg-accent font-medium",
                  )}
                >
                  {text}
                </button>
              ))}
            </fieldset>
            {showTolerance ? (
              <ToleranceSimilarity
                protocolId={protocolId}
                scorecard={scorecard}
                tolerance={tolerance}
              />
            ) : (
              <>
                <SimilarityCurve
                  label={label}
                  series={[
                    {
                      label: "Average error",
                      points: bins.map((bin) => ({
                        ...bin,
                        detail: `Average error ${bin.value.toPrecision(3)}${scorecard.unit ? ` ${scorecard.unit}` : ""}\n${bin.count} test compounds`,
                      })),
                    },
                  ]}
                />
                <p className="text-sm">
                  {bins.length === 1
                    ? "Only one similarity group is available. Its average error is "
                    : "Average error is "}
                  <strong>
                    <ReadoutValue value={first.value} unit={scorecard.unit} />
                  </strong>
                  {bins.length === 1 ? (
                    ` (${first.count} compounds).`
                  ) : (
                    <>
                      {` in the least-similar group (${first.count} compounds), compared with `}
                      <strong>
                        <ReadoutValue value={last.value} unit={scorecard.unit} />
                      </strong>
                      {` in the most-similar group (${last.count} compounds). `}
                      {first.value === last.value
                        ? "The two groups have the same average error."
                        : `Average error was ${last.value < first.value ? "lower" : "higher"} in the most-similar group. This compares the endpoints; the groups in between may vary.`}
                    </>
                  )}
                </p>
              </>
            )}
            <p className="text-xs text-muted-foreground">
              Average error is also called MAE. R² describes overall performance above; within small
              groups it can change simply because their measured values have different spreads.
              {tolerance == null &&
                " Set your acceptable error in the performance overview to compare the percentage within that tolerance."}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
