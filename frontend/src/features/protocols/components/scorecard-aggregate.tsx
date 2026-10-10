"use client";

import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { aggregateTargets } from "../lib/aggregate";
import { higherIsBetter, signed } from "../lib/verdict";
import { METRIC_DESCRIPTIONS, metricLabel } from "../types";

/**
 * One number for every target at once, and the per-target scores it hides.
 *
 * This is the figure a multi-task benchmark publishes and the figure a reader
 * checks a reproduction against. Without it on this page, a mean over twelve
 * Tox21 endpoints has to be copied out of twelve tabs and averaged by hand,
 * which is how the Tox21 and Polaris case studies were in fact written.
 *
 * The average never appears alone. A single number over twelve endpoints hides
 * the one endpoint the model fails on, so the per-target list is part of the
 * same card, weakest first, and the targets left out of the average are named
 * underneath it rather than quietly dropped.
 */
export function ScorecardAggregate({ scorecards }: { scorecards: ScorecardResponse[] }) {
  const result = aggregateTargets(scorecards);

  if (result.kind === "unavailable")
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">No average across targets</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{result.message}</p>
          <p className="mt-2 text-sm text-muted-foreground">
            Each target is scored on its own tab.
          </p>
        </CardContent>
      </Card>
    );

  const { metric, unit, mean, baselineMean, scored, skipped } = result;
  const label = metricLabel(metric);
  const difference = baselineMean == null ? null : mean - baselineMean;
  const better = difference === null ? null : higherIsBetter(metric) === difference > 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">
            Average {label} across {scored.length} target{scored.length === 1 ? "" : "s"}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Every target counts once, however many compounds carry a label for it. This is the
            figure published multi-task results report, so it is the one to compare against them.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-10 gap-y-4">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                This model
              </p>
              <ReadoutValue
                value={mean}
                unit={unit}
                className="text-3xl font-semibold tracking-tight"
              />
            </div>
            {baselineMean != null && (
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Comparison model
                </p>
                <ReadoutValue value={baselineMean} unit={unit} className="text-xl font-semibold" />
              </div>
            )}
            {difference != null && (
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Difference
                </p>
                <span className="text-xl font-semibold tabular-nums">{signed(difference)}</span>
                <span className="ml-2 text-sm text-muted-foreground">
                  {difference === 0 ? "the same" : better ? "in this model's favor" : "against it"}
                </span>
              </div>
            )}
          </div>
          {/* The honest limit of this card. Each per-target tab tests its own
              difference with a paired bootstrap; a difference of two averages
              has no such interval, and printing one here would mean resampling
              the twelve targets' draws jointly, which this does not do. */}
          {difference != null && (
            <p className="border-t pt-3 text-sm text-muted-foreground">
              This is the difference between the two averages on this test set. It carries no
              interval and is not a test of significance: each target's own tab tests its own
              difference.
            </p>
          )}
          {METRIC_DESCRIPTIONS[metric] && (
            <p className="text-sm text-muted-foreground">
              {METRIC_DESCRIPTIONS[metric]} {higherIsBetter(metric) ? "Higher" : "Lower"} is better.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Each target on its own</CardTitle>
          <p className="text-sm text-muted-foreground">
            Weakest first. An average is only as useful as its worst target, and this is where to
            find it.
          </p>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th scope="col" className="pb-2 pr-4 font-medium">
                  Target
                </th>
                <th scope="col" className="pb-2 pr-4 text-right font-medium">
                  {label}
                </th>
                {baselineMean != null && (
                  <th scope="col" className="pb-2 text-right font-medium">
                    Comparison
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {scored.map((entry) => (
                <tr key={entry.target} className="border-b last:border-0">
                  <th scope="row" className="py-2 pr-4 text-left font-mono font-normal">
                    {entry.target}
                  </th>
                  <td className="py-2 pr-4 text-right">
                    <ReadoutValue value={entry.value} unit={unit} />
                  </td>
                  {baselineMean != null && (
                    <td className="py-2 text-right text-muted-foreground">
                      <ReadoutValue value={entry.baseline} unit={unit} />
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {skipped.length > 0 && (
            <div className="mt-4 border-t pt-3">
              <p className="text-sm">
                <span className="font-medium">
                  {skipped.length} target{skipped.length === 1 ? "" : "s"} left out of the average
                </span>{" "}
                — the average above is over {scored.length} of {scored.length + skipped.length}.
              </p>
              <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                {skipped.map((entry) => (
                  <li key={entry.target}>
                    <span className="font-mono">{entry.target}</span>: {entry.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
