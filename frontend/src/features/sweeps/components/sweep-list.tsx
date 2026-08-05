"use client";

import { RUN_STATUS_COPY } from "@/features/runs";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Plus } from "lucide-react";
import Link from "next/link";
// Deep import, not the feature barrel: `index.ts` re-exports this component,
// so importing from it here would be a cycle.
import { useSweeps } from "../hooks/use-sweeps";
import type { Sweep } from "../types";

function SweepRow({ sweep }: { sweep: Sweep }) {
  return (
    <Link
      href={`/sweeps/${sweep.sweep_id}`}
      className="flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
    >
      <div className="flex items-start justify-between gap-3">
        <span className="font-medium">{sweep.name ?? "Sweep"}</span>
        <span className="shrink-0 text-xs text-muted-foreground">
          {sweep.total} run{sweep.total === 1 ? "" : "s"}
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {Object.entries(sweep.by_status)
          .filter(([, count]) => count > 0)
          .map(([status, count]) => (
            <Badge key={status} variant="outline" className="font-normal">
              {count} {RUN_STATUS_COPY[status] ?? status}
            </Badge>
          ))}
      </div>
      <span className="text-xs text-muted-foreground">
        {new Date(sweep.created_at).toLocaleString()}
      </span>
    </Link>
  );
}

export function SweepList() {
  const { data, isLoading, isError } = useSweeps();
  const items = data?.items ?? [];

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Sweeps</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Compare engines and conditions on the same dataset, ranked by score.
          </p>
        </div>
        <Button asChild>
          <Link href="/sweeps/new">
            <Plus className="size-4" />
            New sweep
          </Link>
        </Button>
      </div>

      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load sweeps</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No sweeps yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Start one to compare engines and conditions on the same dataset.
          </p>
          <Button asChild className="mt-4">
            <Link href="/sweeps/new">
              <Plus className="size-4" />
              New sweep
            </Link>
          </Button>
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((sweep) => (
            <SweepRow key={sweep.sweep_id} sweep={sweep} />
          ))}
        </div>
      )}
    </div>
  );
}
