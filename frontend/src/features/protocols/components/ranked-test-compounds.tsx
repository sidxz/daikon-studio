"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { useId, useState } from "react";
import { ScorecardSection } from "./scorecard-section";

export function RankedTestCompounds({ scorecard }: { scorecard: ScorecardResponse }) {
  const binary = scorecard.prediction_kind === "probability";
  const id = useId();
  const [choice, setChoice] = useState<string | null>(null);
  const direction = choice ?? (binary ? "high" : scorecard.direction);
  const high = scorecard.ranked_high ?? [];
  const low = scorecard.ranked_low ?? [];
  if (high.length === 0 && low.length === 0) return null;
  const rows = direction === "high" ? high : direction === "low" ? low : [];
  const summary = scorecard.classification_summary;
  const desiredClass = direction === "low" ? 0 : 1;
  const desiredName = desiredClass === 1 ? "active" : "inactive";
  const hits = rows.filter((row) => row.actual === desiredClass).length;
  const populationHits = summary
    ? desiredClass === 1
      ? summary.true_positive + summary.false_negative
      : summary.true_negative + summary.false_positive
    : null;
  const expected =
    populationHits != null && scorecard.test_count > 0
      ? (populationHits / scorecard.test_count) * rows.length
      : null;

  return (
    <ScorecardSection
      title="Top-ranked test compounds"
      description="Which compounds would the model put first? Ranked using predictions alone; measured results show how those choices turned out."
    >
      <div className="rounded-lg border bg-card">
        <div className="space-y-3 border-b p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm font-medium">
              {rows.length > 0
                ? `Top ${rows.length} of ${scorecard.test_count.toLocaleString()} test compounds`
                : "Choose what you want to find"}
            </p>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <label htmlFor={id}>Ranking goal</label>
              <select
                id={id}
                value={direction ?? ""}
                onChange={(event) => setChoice(event.target.value)}
                className="h-9 max-w-full rounded-md border bg-background px-3 text-sm"
              >
                <option value="" disabled>
                  Choose a goal
                </option>
                <option value="high">{binary ? "Find actives" : "Find higher values"}</option>
                <option value="low">{binary ? "Find inactives" : "Find lower values"}</option>
              </select>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            {binary
              ? `Highest predicted chance of being ${desiredName} first. Active means class 1; inactive means class 0.`
              : direction
                ? `${direction === "high" ? "Highest" : "Lowest"} predicted ${scorecard.target} first${scorecard.unit ? ` (${scorecard.unit})` : ""}.`
                : "This target has no saved preference for higher or lower values. Select a goal to see the ranking."}{" "}
            Ties keep their original test-set order. These compounds already have measured results;
            this is a check of past performance.
          </p>
          {binary && rows.length > 0 && (
            <p className="text-sm" aria-live="polite">
              <strong>
                {hits} of {rows.length}
              </strong>{" "}
              were actually {desiredName}
              {` (${((hits / rows.length) * 100).toFixed(1)}%).`}
              {expected != null && (
                <>
                  {" "}
                  A random selection of {rows.length} from this test set would contain about{" "}
                  {expected.toFixed(1)} {desiredName} compounds on average.
                </>
              )}
            </p>
          )}
        </div>
        {rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">Test compounds ranked by model prediction</caption>
              <thead className="border-b bg-muted/30 text-left text-xs text-muted-foreground">
                <tr>
                  <th scope="col" className="p-3 font-medium">
                    Rank
                  </th>
                  <th scope="col" className="p-3 font-medium">
                    Compound
                  </th>
                  <th scope="col" className="p-3 font-medium">
                    {binary ? `Predicted chance of ${desiredName}` : "Predicted value"}
                  </th>
                  <th scope="col" className="p-3 font-medium">
                    Measured result
                  </th>
                  <th scope="col" className="p-3 font-medium">
                    Similarity to training
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={row.test_index} className="border-b last:border-0">
                    <td className="p-3 tabular-nums text-muted-foreground">{index + 1}</td>
                    <th scope="row" className="p-3 text-left font-normal">
                      <div className="flex items-center gap-3">
                        <StructureThumbnail smiles={row.structure} size={48} className="shrink-0" />
                        <div className="min-w-32 max-w-56">
                          <p className="break-all font-medium">
                            {row.compound_id ?? `Test compound ${row.test_index + 1}`}
                          </p>
                          <p // A 286-residue sequence breaks into a dozen lines and buries the numbers the
                            // row exists to show. Two lines is still the whole of most SMILES.
                            className="mt-1 line-clamp-2 break-all font-mono text-xs text-muted-foreground"
                          >
                            {row.structure}
                          </p>
                        </div>
                      </div>
                    </th>
                    <td className="p-3 tabular-nums">
                      {binary ? (
                        `${((desiredClass === 1 ? row.predicted : 1 - row.predicted) * 100).toFixed(1)}%`
                      ) : (
                        <ReadoutValue value={row.predicted} unit={scorecard.unit} />
                      )}
                    </td>
                    <td className="p-3">
                      {binary ? (
                        row.actual === 1 ? (
                          "Active (1)"
                        ) : (
                          "Inactive (0)"
                        )
                      ) : (
                        <ReadoutValue value={row.actual} unit={scorecard.unit} />
                      )}
                    </td>
                    <td className="p-3 tabular-nums">
                      {row.similarity == null ? "Unavailable" : row.similarity.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </ScorecardSection>
  );
}
