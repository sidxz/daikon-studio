"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import type { PreviewSummary } from "../lib/parse-preview";

/**
 * What was actually read out of the dropped file.
 *
 * The point is to answer "is this the right column?" before a run is
 * submitted rather than after: the structures render from the column that is
 * currently selected, so picking the wrong one looks wrong immediately.
 */
export function PredictionPreview({
  summary,
  column,
}: {
  summary: PreviewSummary;
  column: string;
}) {
  const usable = summary.total - summary.blank;

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3">
      <p className="text-sm">
        <span className="font-medium">
          {usable} compound{usable === 1 ? "" : "s"}
        </span>{" "}
        <span className="text-muted-foreground">
          in <span className="font-mono text-xs">{column}</span>
        </span>
      </p>

      {summary.blank > 0 && (
        // A forecast about what will happen, so it sits outside any button:
        // the worker drops rows it cannot read and scores the rest, and this
        // is the only place that is visible before submitting.
        <p className="mt-1 text-xs text-warning">
          {summary.blank} row{summary.blank === 1 ? " has" : "s have"} no structure and will be
          skipped.
        </p>
      )}

      {summary.sample.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {summary.sample.map((smiles) => (
            <StructureThumbnail key={smiles} smiles={smiles} size={56} />
          ))}
        </div>
      )}
    </div>
  );
}
