"use client";

import { RUN_STATUS_COPY, isTerminal } from "@/features/runs";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
// Deep imports, not the feature barrel: `index.ts` re-exports this component,
// so importing from it here would be a cycle.
import { useCancelSweep, useSweep } from "../hooks/use-sweeps";
import { baselineDelta, formatMetric, rankRuns } from "../lib/rank";
import type { SweepRun } from "../types";

function SweepDetailSkeleton() {
  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

/**
 * Compact `key=value` rendering of a run's conditions -- the engine id alone
 * is ambiguous when two configs share an engine and differ only here, e.g.
 * `n_estimators=300`. Nested values are stringified rather than expanded;
 * sweep conditions are one level deep in practice.
 */
function formatConditions(conditions: SweepRun["conditions"]): string {
  return Object.entries(conditions)
    .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`)
    .join(", ");
}

export function SweepDetail({ id }: { id: string }) {
  const { data: sweep, isPending } = useSweep(id);
  const cancel = useCancelSweep();

  useBreadcrumbTrail(
    sweep ? [{ label: "Sweeps", href: "/sweeps" }, { label: sweep.name ?? "Sweep" }] : null,
  );

  if (isPending || !sweep) return <SweepDetailSkeleton />;

  const ranked = rankRuns(sweep.runs);
  const live = sweep.runs.filter((run) => !isTerminal(run.status));

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">{sweep.name ?? "Sweep"}</h1>
          {/*
            The per-workspace cap (STUDIO_WORKSPACE_MAX_ACTIVE_RUNS, default 10)
            means a large sweep drains in waves. Saying so is the difference
            between a fairness predicate doing its job and a product that looks
            broken. Only shown while it can actually bite.
          */}
          {live.length > 10 && (
            <p className="mt-1 text-sm text-muted-foreground">
              A workspace runs 10 at a time, so this sweep finishes in waves.
            </p>
          )}
        </div>
        {live.length > 0 && (
          <Button variant="outline" disabled={cancel.isPending} onClick={() => cancel.mutate(id)}>
            Cancel sweep
          </Button>
        )}
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>#</TableHead>
              <TableHead>Config</TableHead>
              <TableHead>Engine</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Score</TableHead>
              <TableHead>vs baseline</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ranked.map((run, index) => {
              const delta = baselineDelta(run.metrics);
              const conditions = formatConditions(run.conditions);
              return (
                <TableRow key={run.id}>
                  {/* Rank, not submission index -- the table is sorted, and
                      numbering it by position is the whole point. */}
                  <TableCell className="text-muted-foreground">
                    {run.metrics?.value == null ? "—" : index + 1}
                  </TableCell>
                  <TableCell>
                    <div className="font-medium">
                      {run.protocol_id ? (
                        <Link href={`/protocols/${run.protocol_id}`} className="hover:underline">
                          {run.name}
                        </Link>
                      ) : (
                        run.name
                      )}
                    </div>
                    {conditions && (
                      <div className="font-mono text-xs text-muted-foreground">{conditions}</div>
                    )}
                  </TableCell>
                  <TableCell>{run.engine_id}</TableCell>
                  <TableCell>
                    <Badge
                      variant={run.status === "ready" ? "default" : "outline"}
                      className="font-normal"
                    >
                      {RUN_STATUS_COPY[run.status] ?? run.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{formatMetric(run.metrics)}</TableCell>
                  <TableCell>
                    {delta === null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
