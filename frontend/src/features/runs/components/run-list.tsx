"use client";

import { useProtocols } from "@/features/protocols";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import type { PredictionCountsWire } from "@/shared/lib/api/model";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useRuns } from "../hooks/use-runs";
import { groupRuns } from "../lib/group-runs";
import { RUN_STATUS_COPY, type Run } from "../types";

export function RunList() {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Run[]>([]);
  const { data, isLoading, isError } = useRuns("prediction", cursor);
  const protocols = useProtocols(undefined, 200);
  const memberName = useMemberName();

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  // Resolve the protocol's name so a run is never identified by an id.
  const nameFor = (protocolId: string | null | undefined) =>
    protocols.data?.items.find((protocol) => protocol.id === protocolId)?.name ?? "Prediction run";

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Prediction runs apply a published protocol to a set of compounds. Training runs are
            listed on their protocol.
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
        <div className="space-y-6">
          {groupRuns(items).map((group) => {
            const total = group.days.reduce((sum, day) => sum + day.runs.length, 0);
            return (
              <section key={group.protocolId ?? "none"} className="space-y-3">
                <div className="flex items-baseline justify-between gap-3 border-b border-border pb-1.5">
                  <h2 className="font-medium">
                    {group.protocolId ? (
                      <Link href={`/protocols/${group.protocolId}`} className="hover:underline">
                        {nameFor(group.protocolId)}
                      </Link>
                    ) : (
                      nameFor(group.protocolId)
                    )}
                  </h2>
                  <span className="text-xs text-muted-foreground">
                    {total} run{total === 1 ? "" : "s"}
                  </span>
                </div>
                {group.days.map((day) => (
                  <div key={day.day} className="space-y-1.5">
                    <p className="text-xs font-medium text-muted-foreground">{day.day}</p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {day.runs.map((run) => {
                        const scored = (run.metrics as Partial<PredictionCountsWire> | null)
                          ?.scored_rows;
                        return (
                          <Link
                            key={run.id}
                            href={`/runs/${run.id}`}
                            className="flex items-center justify-between gap-3 rounded-lg border border-border px-4 py-2.5 transition-colors hover:bg-muted/40"
                          >
                            <span className="text-sm tabular-nums">
                              {new Date(run.created_at).toLocaleTimeString("en-US", {
                                hour: "numeric",
                                minute: "2-digit",
                              })}
                              {scored != null && (
                                <span className="text-muted-foreground">
                                  {" "}
                                  · {scored.toLocaleString("en-US")} compound
                                  {scored === 1 ? "" : "s"}
                                </span>
                              )}
                              {run.source && (
                                <span className="text-muted-foreground"> · ChemCellar</span>
                              )}
                            </span>
                            <span className="ml-auto truncate text-xs text-muted-foreground">
                              {memberName(run.requested_by)}
                            </span>
                            <Badge
                              variant={run.status === "ready" ? "default" : "outline"}
                              className="shrink-0 font-normal"
                            >
                              {RUN_STATUS_COPY[run.status] ?? run.status}
                            </Badge>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </section>
            );
          })}
        </div>
      )}

      {data?.next_cursor && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            onClick={() => {
              setPages(items);
              setCursor(data.next_cursor ?? undefined);
            }}
          >
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}
