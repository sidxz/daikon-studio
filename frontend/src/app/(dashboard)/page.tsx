"use client";

import { DatasetCard } from "@/features/datasets/components/dataset-list";
import { useDatasets } from "@/features/datasets/hooks/use-datasets";
import { useProtocols } from "@/features/protocols/hooks/use-protocols";
import { useRuns } from "@/features/runs/hooks/use-runs";
import { ItemCard } from "@/shared/components/item-card";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { StatusDot } from "@/shared/components/status-dot";
import { Button } from "@/shared/components/ui/button";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { targetsOf } from "@/shared/lib/targets";
import { Plus } from "lucide-react";
import Link from "next/link";

function LoadingCards() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {[0, 1, 2].map((key) => (
        <Skeleton key={key} className="h-36 w-full" />
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const datasets = useDatasets(undefined, 3);
  const protocols = useProtocols(undefined, 3);
  const memberName = useMemberName();
  const active = useRuns(null, undefined, { mine: true, statuses: ["pending", "running"] });
  const live = (active.data?.items ?? []).filter(
    (run) => run.status === "pending" || run.status === "running",
  );

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Dashboard"
        description="Recent datasets, protocols, and active runs."
        action={
          <>
            <Button asChild variant="outline" size="sm">
              <Link href="/datasets/new">
                <Plus className="size-3.5" />
                New dataset
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link href="/protocols/new">Train a protocol</Link>
            </Button>
            <Button asChild size="sm">
              <Link href="/runs/new">Run predictions</Link>
            </Button>
          </>
        }
      />
      {active.isError && (
        <QueryError
          title="Could not load active runs"
          retry={() => active.refetch()}
          retrying={active.isFetching}
        />
      )}
      {live.length > 0 && (
        <section aria-label="Work in progress" className="space-y-3">
          <h2 className="text-base">Work in progress</h2>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {live.slice(0, 6).map((run) => (
              <Link
                key={run.id}
                href={`/runs/${run.id}`}
                className="rounded-md border border-border bg-card p-3 transition-colors hover:border-primary/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <div className="flex items-center gap-2">
                  <StatusDot tone="active" />
                  <span className="truncate text-sm font-medium">
                    {run.name ?? (run.kind === "training" ? "Training run" : "Prediction run")}
                  </span>
                </div>
                <div className="mt-3 flex items-center gap-3">
                  <p className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
                    {run.phase ?? (run.status === "pending" ? "Queued" : "Running")}
                  </p>
                  <Progress
                    className="w-16 shrink-0"
                    value={Math.round(run.progress * 100)}
                    aria-label="Run progress"
                  />
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}
      <section aria-labelledby="recent-datasets" className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h2 id="recent-datasets" className="text-base">
            Recent datasets
          </h2>
          <Link href="/datasets" className="text-xs text-primary hover:underline">
            View all →
          </Link>
        </div>
        {datasets.isLoading && <LoadingCards />}
        {datasets.isError && (
          <QueryError
            title="Could not load datasets"
            retry={() => datasets.refetch()}
            retrying={datasets.isFetching}
          />
        )}
        {datasets.data?.items.length === 0 && (
          <div className="rounded-md border border-dashed border-border p-5">
            <p className="text-sm">No datasets yet</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Upload a CSV of structures and measurements to start training.
            </p>
            <Link
              href="/datasets/new"
              className="mt-3 inline-block text-xs font-medium text-primary hover:underline"
            >
              Upload a dataset →
            </Link>
          </div>
        )}
        <div className="grid items-stretch gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {datasets.data?.items.map((dataset) => (
            <DatasetCard key={dataset.id} dataset={dataset} draggable={false} />
          ))}
        </div>
      </section>
      <section aria-labelledby="recent-protocols" className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h2 id="recent-protocols" className="text-base">
            Recent protocols
          </h2>
          <Link href="/protocols" className="text-xs text-primary hover:underline">
            View all →
          </Link>
        </div>
        {protocols.isLoading && <LoadingCards />}
        {protocols.isError && (
          <QueryError
            title="Could not load protocols"
            retry={() => protocols.refetch()}
            retrying={protocols.isFetching}
          />
        )}
        {protocols.data?.items.length === 0 && (
          <div className="rounded-md border border-dashed border-border p-5">
            <p className="text-sm">No protocols yet</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Train a model on a dataset, then review its scorecard.
            </p>
            <Link
              href="/protocols/new"
              className="mt-3 inline-block text-xs font-medium text-primary hover:underline"
            >
              Train a protocol →
            </Link>
          </div>
        )}
        <div className="grid items-stretch gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {protocols.data?.items.map((protocol) => (
            <ItemCard
              key={protocol.id}
              href={`/protocols/${protocol.id}`}
              name={protocol.name}
              subtitle={`Predicts ${targetsOf(protocol.readouts ?? []).join(", ") || "N/A"}`}
              draft={protocol.status === "draft"}
              footerStart={`Version ${protocol.protocol_version}`}
              creator={memberName(protocol.created_by)}
              creatorId={protocol.created_by}
              createdAt={protocol.created_at}
            >
              <p className="text-xs text-muted-foreground">
                {protocol.status === "draft"
                  ? "Review scorecard before publishing"
                  : "Available for prediction"}
              </p>
            </ItemCard>
          ))}
        </div>
      </section>
    </div>
  );
}
