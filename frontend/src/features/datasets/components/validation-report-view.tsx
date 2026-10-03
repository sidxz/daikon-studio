"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import type { ValidationReport } from "../types";

function Stat({
  value,
  label,
  tone = "default",
}: {
  value: string | number;
  label: string;
  tone?: "default" | "warn" | "bad";
}) {
  const valueTone =
    tone === "bad" ? "text-destructive" : tone === "warn" ? "text-warning" : "text-foreground";
  return (
    <div className="flex h-full flex-col justify-center rounded-lg border border-border p-4">
      <span className={`text-2xl font-semibold tabular-nums ${valueTone}`}>{value}</span>
      <span className="mt-0.5 text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

/**
 * The full report, whether the file was accepted or refused.
 *
 * A rejection arrives as a 422 whose body *is* this object. Collapsing that
 * into a toast would throw away the only thing that lets a scientist fix a
 * ten-thousand-row file: which rows failed, and why. Row numbers are 1-based
 * positions in the file as uploaded, so they can be looked up directly.
 */
export function ValidationReportView({
  report,
  rejected = false,
}: {
  report: ValidationReport;
  rejected?: boolean;
}) {
  const hasInvalid = report.invalid.length > 0;
  const hasConflicts = report.conflicting.length > 0;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat value={report.total_rows.toLocaleString()} label="Rows in the file" />
        {/* `valid_rows` counts rows whose structure parsed, which is *before*
            replicates are grouped -- so it is not the compound count. Deriving
            it here is what stops the page saying "1,008 usable compounds" for a
            dataset that trains on 997. */}
        <Stat
          value={(report.valid_rows - report.duplicates_collapsed).toLocaleString()}
          label="Compounds to train on"
        />
        <Stat
          value={report.duplicates_collapsed.toLocaleString()}
          label="Duplicates collapsed"
          tone={report.duplicates_collapsed > 0 ? "warn" : "default"}
        />
        <Stat
          value={report.salts_flagged.toLocaleString()}
          label="Salts or mixtures flagged"
          tone={report.salts_flagged > 0 ? "warn" : "default"}
        />
      </div>

      {report.duplicate_spread != null && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Assay noise floor</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">
              Repeat measurements of the same compound disagreed by{" "}
              <span className="font-mono font-medium">{report.duplicate_spread.toFixed(3)}</span> on
              average.
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              That spread is free information: no model trained on this data can honestly beat it,
              so it becomes the floor your scorecard is measured against.
            </p>
          </CardContent>
        </Card>
      )}

      {hasInvalid && (
        <Card className={rejected ? "border-destructive/50" : undefined}>
          <CardHeader>
            <CardTitle className="text-base">
              {report.invalid.length.toLocaleString()} row
              {report.invalid.length === 1 ? "" : "s"} could not be used
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Structures that did not parse, and target values that are empty, not a number, or not
              0/1. Rejected rather than dropped quietly. Row numbers are positions in the file you
              uploaded.
            </p>
          </CardHeader>
          <CardContent className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="pb-2 pr-4 font-medium">Row</th>
                  <th className="pb-2 pr-4 font-medium">Value</th>
                  <th className="pb-2 font-medium">Why</th>
                </tr>
              </thead>
              <tbody>
                {report.invalid.map((row) => (
                  <tr key={`${row.row_number}-${row.value}`} className="border-b last:border-0">
                    <td className="py-1.5 pr-4 font-mono tabular-nums">{row.row_number}</td>
                    <td className="py-1.5 pr-4 font-mono text-xs break-all">{row.value}</td>
                    <td className="py-1.5 text-muted-foreground">{row.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {hasConflicts && (
        <Card className={rejected ? "border-destructive/50" : undefined}>
          <CardHeader>
            <CardTitle className="text-base">
              {report.conflicting.length.toLocaleString()} compound
              {report.conflicting.length === 1 ? "" : "s"} labelled both ways
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              The same structure appears as active and inactive. That is a decision about your data,
              not one a majority vote should make silently — so the file is refused until you
              resolve it.
            </p>
          </CardHeader>
          <CardContent className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="pb-2 pr-4 font-medium">Rows</th>
                  <th className="pb-2 pr-4 font-medium">Labels</th>
                  <th className="pb-2 font-medium">Structure</th>
                </tr>
              </thead>
              <tbody>
                {report.conflicting.map((row) => (
                  <tr key={row.structure} className="border-b last:border-0">
                    <td className="py-1.5 pr-4 font-mono tabular-nums">
                      {row.row_numbers.join(", ")}
                    </td>
                    <td className="py-1.5 pr-4 font-mono">{row.values.join(" vs ")}</td>
                    <td className="py-1.5 font-mono text-xs break-all">{row.structure}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
