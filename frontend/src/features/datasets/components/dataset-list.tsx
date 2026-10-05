"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useDatasets } from "../hooks/use-datasets";
import { type Dataset, SPLIT_COPY } from "../types";

function DatasetRow({ dataset }: { dataset: Dataset }) {
  const creator = useMemberName()(dataset.created_by);
  return (
    <Link
      href={`/datasets/${dataset.id}`}
      className="flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
    >
      <div className="flex items-start justify-between gap-3">
        <span className="font-medium">{dataset.name}</span>
        <Badge variant="outline" className="shrink-0 font-normal">
          {SPLIT_COPY[dataset.split.strategy].title}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>{dataset.row_count.toLocaleString()} compounds</span>
        <span className="font-mono">
          {dataset.targets.map((target) => target.column).join(", ")}
        </span>
        {dataset.targets.length === 1 && dataset.targets[0].unit && (
          <span className="font-mono">{dataset.targets[0].unit}</span>
        )}
        <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
        {creator && <span>by {creator}</span>}
      </div>
    </Link>
  );
}

export function DatasetList() {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Dataset[]>([]);
  const { data, isLoading, isError, error } = useDatasets(cursor);

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Datasets</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            An immutable snapshot of structures and one or more targets, with a fixed split, so
            protocols trained on it are reproducible.
          </p>
        </div>
        <Button asChild>
          <Link href="/datasets/new">
            <Plus className="size-4" />
            New dataset
          </Link>
        </Button>
      </div>

      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load datasets</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No datasets yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload a CSV of structures and measurements to get started.
          </p>
          <Button asChild className="mt-4">
            <Link href="/datasets/new">
              <Plus className="size-4" />
              New dataset
            </Link>
          </Button>
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((dataset) => (
            <DatasetRow key={dataset.id} dataset={dataset} />
          ))}
        </div>
      )}

      {/* `total_count` is always null in this API, so there is no "1-50 of 320"
          to render honestly -- load-more is the truthful affordance. */}
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
