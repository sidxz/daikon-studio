"use client";

import type { DatasetReadinessResponse } from "@/shared/lib/api/model";

const PARTITIONS = [
  { key: "train", label: "Training", color: "bg-primary" },
  { key: "validation", label: "Validation", color: "bg-primary/50" },
  { key: "test", label: "Test", color: "bg-primary/25" },
] as const;

export function DatasetReadinessView({ readiness }: { readiness: DatasetReadinessResponse }) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex h-3 overflow-hidden rounded-full bg-muted" aria-hidden="true">
          {PARTITIONS.map(({ key, color }) => (
            <div
              key={key}
              className={color}
              style={{
                width: `${(100 * (readiness.partition_counts[key] ?? 0)) / Math.max(readiness.row_count, 1)}%`,
              }}
            />
          ))}
        </div>
        <dl className="grid grid-cols-3 gap-3">
          {PARTITIONS.map(({ key, label, color }) => (
            <div key={key}>
              <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className={`size-2 rounded-full ${color}`} />
                {label}
              </dt>
              <dd className="mt-1 text-sm font-medium tabular-nums">
                {(readiness.partition_counts[key] ?? 0).toLocaleString()}{" "}
                <span className="font-normal text-muted-foreground">
                  (
                  {Math.round(
                    (100 * (readiness.partition_counts[key] ?? 0)) /
                      Math.max(readiness.row_count, 1),
                  )}
                  %)
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </div>
      {readiness.class_balance.length > 0 && (
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-xs">
            <caption className="px-3 py-2 text-left font-medium">
              Class balance after preparation
            </caption>
            <thead className="bg-muted/40 text-left">
              <tr>
                <th className="px-3 py-2">Target</th>
                <th className="px-3 py-2">Set</th>
                <th className="px-3 py-2">Active</th>
                <th className="px-3 py-2">Inactive</th>
              </tr>
            </thead>
            <tbody>
              {readiness.class_balance.map((entry) => (
                <tr key={`${entry.column}-${entry.split}`} className="border-t">
                  <td className="px-3 py-2 font-mono">{entry.column}</td>
                  <td className="px-3 py-2 capitalize">{entry.split}</td>
                  <td className="px-3 py-2 tabular-nums">{entry.positive.toLocaleString()}</td>
                  <td className="px-3 py-2 tabular-nums">{entry.negative.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {readiness.warnings.length > 0 && (
        <div
          className="space-y-2 rounded-lg border border-warning/40 bg-warning/5 p-3"
          aria-live="polite"
        >
          <p className="text-sm font-medium">Things to consider</p>
          <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
            {readiness.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
          <p className="text-xs text-muted-foreground">
            These are warnings. You can continue with this split or review your data and split
            choices.
          </p>
        </div>
      )}
    </div>
  );
}
