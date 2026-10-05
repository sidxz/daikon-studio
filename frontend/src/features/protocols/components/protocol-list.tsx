"use client";

import { useEngines } from "@/features/engines";
import { FolderRail } from "@/features/folders";
import { ProtocolsIcon } from "@/shared/components/icons/nav-icons";
import { SegmentedToggle } from "@/shared/components/segmented-toggle";
import { StatusDot } from "@/shared/components/status-dot";
import { Button } from "@/shared/components/ui/button";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useActiveTrainingRuns, useProtocols, useTrainingHeadlines } from "../hooks/use-protocols";
import type { Protocol } from "../types";
import { ProtocolCard } from "./protocol-card";

/** Runs that have not produced a Protocol yet. Lighter than the grid below: it is a status, not a result. */
function InTraining() {
  const runs = useActiveTrainingRuns();
  if (runs.length === 0) return null;
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">In training</h2>
      <ul className="divide-y divide-border rounded-lg border border-border">
        {runs.map((run) => (
          <li key={run.id}>
            <Link
              href={`/runs/${run.id}`}
              className="flex items-center gap-3 px-3 py-2 text-sm transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
            >
              <StatusDot tone="active" />
              <span className="min-w-0 flex-1 truncate">
                <span className="font-medium">{run.name ?? "Training run"}</span>
                <span className="ml-2 text-muted-foreground">{run.phase ?? "Queued"}</span>
              </span>
              <Progress
                value={Math.round(run.progress * 100)}
                aria-label="Training progress"
                className="w-20 shrink-0 sm:w-28"
              />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ProtocolGrid({ mine, folderId }: { mine: boolean; folderId: string | undefined }) {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Protocol[]>([]);
  const { data, isLoading, isError } = useProtocols(cursor, undefined, { mine, folderId });
  const memberName = useMemberName();
  const scores = useTrainingHeadlines();
  const engines = useEngines().data;

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);
  const engineName = (id: string) => engines?.find((engine) => engine.id === id)?.name ?? id;

  return (
    <>
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <Skeleton className="h-44 w-full" />
          <Skeleton className="h-44 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load protocols</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <ProtocolsIcon className="mx-auto mb-3 size-6 text-icon-protocols opacity-60" />
          {folderId ? (
            <p className="text-sm font-medium">
              This folder is empty. Drag a protocol here or use Move to folder.
            </p>
          ) : mine ? (
            <p className="text-sm font-medium">You have not trained a protocol yet.</p>
          ) : (
            <>
              <p className="text-sm font-medium">No protocols yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Freeze a dataset first, then train a model on it.
              </p>
            </>
          )}
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((protocol) => (
            <ProtocolCard
              key={protocol.id}
              protocol={protocol}
              scores={scores.get(protocol.id) ?? []}
              engineName={engineName(protocol.engine_id)}
              creator={memberName(protocol.created_by)}
            />
          ))}
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
    </>
  );
}

export function ProtocolList() {
  const { params, set } = useUrlParams();
  const mine = params.get("mine") === "1";
  const folderId = params.get("folder") ?? undefined;
  const setMine = (next: boolean) => set({ mine: next ? "1" : undefined });

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Protocols</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Trained models, each scored against a baseline. A draft is private to you and workspace
            admins until you publish it.
          </p>
        </div>
        <Button asChild>
          <Link href="/protocols/new">
            <Plus className="size-4" />
            Train a protocol
          </Link>
        </Button>
      </div>

      <div className="flex flex-col gap-4 md:grid md:grid-cols-[13rem_minmax(0,1fr)] md:items-start md:gap-8">
        <FolderRail kind="protocol" activeId={folderId} onSelect={(id) => set({ folder: id })} />

        <div className="min-w-0 space-y-4">
          <SegmentedToggle
            label="Protocol owner"
            options={[
              { value: "all", label: "All" },
              { value: "mine", label: "Mine" },
            ]}
            value={mine ? "mine" : "all"}
            onChange={(value) => setMine(value === "mine")}
          />

          <InTraining />

          <ProtocolGrid key={`${mine}-${folderId}`} mine={mine} folderId={folderId} />
        </div>
      </div>
    </div>
  );
}
