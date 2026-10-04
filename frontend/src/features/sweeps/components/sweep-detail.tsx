"use client";

import { RUN_STATUS_COPY } from "@/features/runs";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { ApiError } from "@/shared/lib/api/custom-instance";
import { isTerminal } from "@/shared/lib/query-defaults";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import { ArrowDown } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
// Deep imports, not the feature barrel: `index.ts` re-exports this component,
// so importing from it here would be a cycle.
import { useCancelSweep, useSweep } from "../hooks/use-sweeps";
import {
  baselineDelta,
  formatMetric,
  headlineFor,
  isRankable,
  sortDirection,
  sortRuns,
  sweepTargets,
} from "../lib/rank";
import type { SweepRun } from "../types";

function SweepDetailSkeleton() {
  return (
    <div className="mx-auto w-full max-w-7xl space-y-4 p-2">
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
  const [sortBy, setSortBy] = useState<string | null>(null);
  // isLoadingError, not isError: a failed background refetch keeps the loaded
  // sweep on screen (see run-detail for the same reasoning).
  const { data: sweep, isLoadingError, error, refetch } = useSweep(id);
  const cancel = useCancelSweep();

  useBreadcrumbTrail(
    sweep ? [{ label: "Sweeps", href: "/sweeps" }, { label: sweep.name ?? "Sweep" }] : null,
  );

  // Polling stops on error, so a skeleton here would never resolve. The
  // server's message is not shown -- a 404's names the sweep by its UUID.
  if (isLoadingError) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <div className="mx-auto w-full max-w-7xl p-2">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">
            {missing ? "This sweep does not exist in this workspace" : "Could not load this sweep"}
          </p>
          {!missing && (
            <Button variant="outline" size="sm" className="mt-3" onClick={() => refetch()}>
              Try again
            </Button>
          )}
        </div>
      </div>
    );
  }

  if (!sweep) return <SweepDetailSkeleton />;

  const targets = sweepTargets(sweep.runs);
  // One target sorts by default, so a single-target sweep reads exactly as it
  // always has; several wait for a click (see `sortRuns`).
  const active = sortBy ?? (targets.length === 1 ? targets[0] : null);
  const ordered = sortRuns(sweep.runs, active);
  const live = sweep.runs.filter((run) => !isTerminal(run.status));

  return (
    <div className="mx-auto w-full max-w-7xl space-y-4 p-2">
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
              Each workspace runs at most 10 runs concurrently; the rest are queued.
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
              <TableHead>Configuration</TableHead>
              <TableHead>Engine</TableHead>
              <TableHead>Status</TableHead>
              {targets.map((column) => (
                <TableHead
                  key={column}
                  aria-sort={active === column ? sortDirection(sweep.runs, column) : undefined}
                >
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 font-mono hover:underline"
                    onClick={() => setSortBy(column)}
                  >
                    {column}
                    {active === column && <ArrowDown className="size-3.5" />}
                  </button>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {ordered.map((run, index) => {
              const conditions = formatConditions(run.conditions);
              return (
                <TableRow key={run.id}>
                  {/* Rank, not submission index -- the table is sorted, and
                      numbering it by position is the whole point. */}
                  <TableCell className="text-muted-foreground">
                    {active && isRankable(headlineFor(run, active)) ? index + 1 : "—"}
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
                    {/* Same idiom as `RunDetail`'s progress card -- phase text
                        over a `Progress` bar -- so a chemprop sweep shows
                        which of ten identical "Running" rows is nearly done. */}
                    {!isTerminal(run.status) && (
                      <div className="mt-1.5 w-28 space-y-0.5 whitespace-normal">
                        <Progress value={Math.round(run.progress * 100)} />
                        <p className="text-xs text-muted-foreground">{run.phase ?? "Starting…"}</p>
                      </div>
                    )}
                  </TableCell>
                  {targets.map((column) => {
                    const headline = headlineFor(run, column);
                    const delta = baselineDelta(headline);
                    return (
                      <TableCell key={column}>
                        <div>{formatMetric(headline)}</div>
                        {delta !== null && (
                          <div className="text-xs text-muted-foreground">
                            {`${delta >= 0 ? "+" : ""}${delta.toFixed(3)} vs baseline`}
                          </div>
                        )}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
