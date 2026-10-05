"use client";

import { FolderStrip, MoveToFolderMenu, setDragItem, useFolders } from "@/features/folders";
import { SegmentedToggle } from "@/shared/components/segmented-toggle";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useActiveTrainingRuns, useProtocols } from "../hooks/use-protocols";
import type { Protocol } from "../types";

/** Runs that have not produced a Protocol yet. Lighter than the grid below: it is a status, not a result. */
function InTraining() {
  const runs = useActiveTrainingRuns();
  if (runs.length === 0) return null;
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">In training</h2>
      <ul className="space-y-2">
        {runs.map((run) => (
          <li key={run.id}>
            <Link
              href={`/runs/${run.id}`}
              className="flex items-center justify-between gap-4 rounded-md border border-border/60 px-3 py-2 text-sm transition-colors hover:bg-muted/40"
            >
              <div className="min-w-0">
                <div className="truncate font-medium">{run.name ?? "Training run"}</div>
                <div className="text-xs text-muted-foreground">{run.phase ?? "Queued"}</div>
              </div>
              <Progress
                value={Math.round(run.progress * 100)}
                aria-label="Training progress"
                className="w-28 shrink-0"
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
  const folders = useFolders("protocol").data;

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <>
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load protocols</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
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
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((protocol) => {
            const creator = memberName(protocol.created_by);
            const folder = folderId
              ? undefined
              : folders?.items.find((candidate) => candidate.id === protocol.folder_id);
            return (
              // A div card with a stretched title link: the menu is a sibling, not nested in the <a>.
              <div
                key={protocol.id}
                draggable={folders?.can_edit ?? false}
                onDragStart={(e) => setDragItem(e, "protocol", protocol.id)}
                className="relative flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
              >
                <div className="flex items-start justify-between gap-3">
                  <Link
                    href={`/protocols/${protocol.id}`}
                    draggable={false}
                    className="font-medium after:absolute after:inset-0"
                  >
                    {protocol.name}
                  </Link>
                  <div className="relative z-10 flex shrink-0 items-center gap-1">
                    <Badge
                      variant={protocol.status === "draft" ? "outline" : "default"}
                      className="font-normal"
                    >
                      {protocol.status === "draft" ? "Draft" : "Published"}
                    </Badge>
                    <MoveToFolderMenu
                      kind="protocol"
                      itemId={protocol.id}
                      currentFolderId={protocol.folder_id}
                    />
                  </div>
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  <span className="font-mono">{protocol.engine_id}</span>
                  <span>v{protocol.protocol_version}</span>
                  <span>{new Date(protocol.created_at).toLocaleDateString()}</span>
                  {creator && <span>by {creator}</span>}
                  {folder && <span>{folder.name}</span>}
                </div>
              </div>
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
    </>
  );
}

export function ProtocolList() {
  const { params, set } = useUrlParams();
  const mine = params.get("mine") === "1";
  const folderId = params.get("folder") ?? undefined;
  const setMine = (next: boolean) => set({ mine: next ? "1" : undefined });

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Protocols</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Trained models, each scored against a baseline. A draft is visible only to you and
            workspace admins; publishing locks it and shares it with the workspace.
          </p>
        </div>
        <Button asChild>
          <Link href="/protocols/new">
            <Plus className="size-4" />
            Train a protocol
          </Link>
        </Button>
      </div>

      <SegmentedToggle
        label="Protocol owner"
        options={[
          { value: "all", label: "All" },
          { value: "mine", label: "Mine" },
        ]}
        value={mine ? "mine" : "all"}
        onChange={(value) => setMine(value === "mine")}
      />

      <FolderStrip kind="protocol" activeId={folderId} onSelect={(id) => set({ folder: id })} />

      <InTraining />

      <ProtocolGrid key={`${mine}-${folderId}`} mine={mine} folderId={folderId} />
    </div>
  );
}
