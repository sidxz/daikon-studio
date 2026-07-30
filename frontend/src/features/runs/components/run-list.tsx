"use client";

import { useProtocols } from "@/features/protocols";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useRuns } from "../hooks/use-runs";
import { RUN_STATUS_COPY } from "../types";

export function RunList() {
  const { data, isLoading, isError } = useRuns("prediction");
  const protocols = useProtocols();
  const items = data?.items ?? [];

  // Resolve the protocol's name so a run is never identified by an id.
  const nameFor = (protocolId: string | null | undefined) =>
    protocols.data?.items.find((protocol) => protocol.id === protocolId)?.name ?? "Prediction run";

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A published protocol scoring a set of compounds. Training runs live on their protocol
            instead.
          </p>
        </div>
        <Button asChild>
          <Link href="/runs/new">
            <Plus className="size-4" />
            Run a protocol
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
          <p className="text-sm font-medium text-destructive">Could not load runs</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No runs yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Publish a protocol, then run it across your own compounds.
          </p>
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((run) => (
            <Link
              key={run.id}
              href={`/runs/${run.id}`}
              className="flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
            >
              <div className="flex items-start justify-between gap-3">
                <span className="font-medium">{nameFor(run.protocol_id)}</span>
                <Badge
                  variant={run.status === "ready" ? "default" : "outline"}
                  className="shrink-0 font-normal"
                >
                  {RUN_STATUS_COPY[run.status] ?? run.status}
                </Badge>
              </div>
              <span className="text-xs text-muted-foreground">
                {new Date(run.created_at).toLocaleString()}
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
