"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { useState } from "react";
import { type WorstOrder, orderWorstRows } from "../lib/worst-rows";

/** One color per series, cycled; literal class names so Tailwind generates them. */
const SERIES_DOT = ["bg-chart-1", "bg-chart-2", "bg-chart-3", "bg-chart-4", "bg-chart-5"];

/**
 * The scorecard's largest prediction errors, as one grid. Compounds that share
 * a scaffold are a series; the order toggle (series side by side, or worst
 * first) appears only when there is a series to group by, since otherwise both
 * orders are the same.
 */
export function LargestErrors({ scorecard }: { scorecard: ScorecardResponse }) {
  const [order, setOrder] = useState<WorstOrder>("series");
  if (scorecard.worst_rows.length === 0) return null;

  const ordered = orderWorstRows(scorecard.worst_rows, order);
  const series = [
    ...new Map(ordered.flatMap(({ series }) => (series ? [[series.label, series]] : []))).values(),
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <CardTitle className="text-base">Largest prediction errors</CardTitle>
          {series.length > 0 && (
            <fieldset className="inline-flex gap-0.5 rounded-lg border bg-card p-0.5">
              <legend className="sr-only">Order</legend>
              {(
                [
                  ["series", "By series"],
                  ["error", "By error"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={order === value}
                  onClick={() => setOrder(value)}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-xs",
                    order === value
                      ? "bg-accent font-medium text-accent-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </fieldset>
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          The {scorecard.worst_rows.length} test compounds with the largest absolute error.
          Compounds that share a Bemis–Murcko scaffold carry the same series tag; several misses in
          one series point to a chemotype the model predicts poorly.{" "}
          {series.length > 0
            ? `Shared scaffolds: ${series.map((tag) => `${tag.label} (${tag.size})`).join(", ")}.`
            : "No two share a scaffold."}
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {ordered.map(({ row, series: tag }) => (
            <div
              key={row.structure}
              className="flex h-full flex-col items-center gap-2 rounded-lg border border-border p-3"
            >
              {/* Fixed height whether or not this card has a tag, so the cards line
                  up; left out entirely when no compound shares a scaffold. */}
              {series.length > 0 && (
                <div className="flex h-5 w-full items-center">
                  {tag && (
                    <span
                      title={tag.scaffold}
                      className="inline-flex items-center gap-1.5 rounded-full border px-2 text-[11px] font-medium"
                    >
                      <span
                        className={cn(
                          "size-2 rounded-full",
                          SERIES_DOT[tag.index % SERIES_DOT.length],
                        )}
                        aria-hidden="true"
                      />
                      Series {tag.label} · {tag.size}
                    </span>
                  )}
                </div>
              )}
              {row.compound_id && (
                <p
                  className="w-full truncate text-center font-mono text-xs font-medium"
                  title={row.compound_id}
                >
                  {row.compound_id}
                </p>
              )}
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
                  <dt className="text-muted-foreground">Abs. error</dt>
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
                      className={row.similarity < 0.3 ? "font-medium text-warning" : "tabular-nums"}
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
      </CardContent>
    </Card>
  );
}
