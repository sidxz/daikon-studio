"use client";

import { useEngines } from "@/features/engines";
import { FolderTabs } from "@/features/folders";
import { ProtocolsIcon } from "@/shared/components/icons/nav-icons";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { StatusDot } from "@/shared/components/status-dot";
import { Button } from "@/shared/components/ui/button";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Plus, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import {
  type ProtocolFilters as FilterValues,
  useActiveTrainingRuns,
  useProtocols,
  useStoppedTrainingRuns,
  useTrainingHeadlines,
} from "../hooks/use-protocols";
import type { Protocol } from "../types";
import { ProtocolCard } from "./protocol-card";
import { ProtocolFilters } from "./protocol-filters";

/** Runs that have not produced a Protocol yet. Lighter than the grid below: it is a status, not a result. */
function InTraining() {
  const runs = useActiveTrainingRuns();
  const stopped = useStoppedTrainingRuns();
  if (runs.length === 0 && stopped.length === 0) return null;
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">In training</h2>
      <ul className="divide-y divide-border rounded-lg border border-border">
        {runs.map((run) => (
          <li key={run.id}>
            <Link href={`/runs/${run.id}`} className={ROW}>
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
        {stopped.map((run) => (
          <li key={run.id}>
            <Link href={`/runs/${run.id}`} className={ROW}>
              <StatusDot tone={run.status === "failed" ? "failed" : "muted"} />
              <span className="min-w-0 flex-1 truncate">
                <span className="font-medium">{run.name ?? "Training run"}</span>
                <span className="ml-2 text-muted-foreground">
                  {run.status === "failed" ? "Failed" : "Canceled"} · open to resume or delete
                </span>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

const ROW =
  "flex items-center gap-3 px-3 py-2 text-sm transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring";

function ProtocolGrid({ filters, onClear }: { filters: FilterValues; onClear: () => void }) {
  const { mine, folderId } = filters;
  const filtered = Boolean(filters.q || filters.engineId || filters.status);
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Protocol[]>([]);
  const { data, isLoading, isError, refetch, isFetching } = useProtocols(
    cursor,
    undefined,
    filters,
  );
  const memberName = useMemberName();
  const scores = useTrainingHeadlines();
  const engines = useEngines().data;

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);
  const engineName = (id: string) => engines?.find((engine) => engine.id === id)?.name ?? id;

  return (
    <>
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      )}

      {isError && (
        <QueryError
          title="Could not load protocols"
          retry={() => refetch()}
          retrying={isFetching}
        />
      )}

      {data && items.length === 0 && filtered && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <Search className="mx-auto mb-3 size-5 text-muted-foreground" aria-hidden />
          <p className="text-sm font-medium">No protocols match these filters.</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try another search or clear the filters.
          </p>
          <Button variant="outline" className="mt-4" onClick={onClear}>
            Clear filters
          </Button>
        </div>
      )}

      {data && items.length === 0 && !filtered && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <ProtocolsIcon className="mx-auto mb-3 size-6 text-icon-protocols opacity-60" />
          {folderId ? (
            <p className="text-sm font-medium">
              This folder is empty. Drag a protocol here or use Move to folder.
            </p>
          ) : mine ? (
            <>
              <p className="text-sm font-medium">You have not trained a protocol yet.</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Choose a dataset to train your first model.
              </p>
              <Button asChild className="mt-4">
                <Link href="/protocols/new">Train a protocol</Link>
              </Button>
            </>
          ) : (
            <>
              <p className="text-sm font-medium">No protocols yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Start with a dataset, then train a model and review its scorecard.
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                <Button asChild>
                  <Link href="/protocols/new">Train a protocol</Link>
                </Button>
                <Button asChild variant="outline">
                  <Link href="/datasets/new">Create a dataset</Link>
                </Button>
              </div>
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
  const status = params.get("status");
  const filters: FilterValues = {
    mine,
    folderId,
    q: params.get("q")?.trim() || undefined,
    engineId: params.get("engine_id") || undefined,
    status: status === "draft" || status === "published" ? status : undefined,
  };
  const clearFilters = () =>
    set({ q: undefined, engine_id: undefined, status: undefined, mine: undefined });

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Protocols"
        description="Trained models and their scorecards. Drafts are visible to you and workspace admins."
        action={
          <Button asChild>
            <Link href="/protocols/new">
              <Plus className="size-4" />
              Train a protocol
            </Link>
          </Button>
        }
      />

      <FolderTabs kind="protocol" activeId={folderId} onSelect={(id) => set({ folder: id })} />
      <ProtocolFilters filters={filters} onChange={set} />
      <InTraining />
      <ProtocolGrid key={JSON.stringify(filters)} filters={filters} onClear={clearFilters} />
    </div>
  );
}
