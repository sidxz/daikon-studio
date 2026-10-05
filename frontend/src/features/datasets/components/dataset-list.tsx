"use client";

import { FolderRail, MoveToFolderMenu, setDragItem, useFolders } from "@/features/folders";
import { DatasetsIcon } from "@/shared/components/icons/nav-icons";
import { ItemCard, LeadNumber } from "@/shared/components/item-card";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useDatasets } from "../hooks/use-datasets";
import { type Dataset, SPLIT_COPY } from "../types";

function DatasetCard({ dataset, draggable }: { dataset: Dataset; draggable: boolean }) {
  const creator = useMemberName()(dataset.created_by);
  const [target, ...others] = dataset.targets;
  return (
    <ItemCard
      href={`/datasets/${dataset.id}`}
      name={dataset.name}
      subtitle={
        target && `Targets ${target.column}${others.length ? ` and ${others.length} more` : ""}`
      }
      action={
        <MoveToFolderMenu kind="dataset" itemId={dataset.id} currentFolderId={dataset.folder_id} />
      }
      footerStart={`${SPLIT_COPY[dataset.split.strategy].title} split`}
      creator={creator}
      createdAt={dataset.created_at}
      draggable={draggable}
      onDragStart={(e) => setDragItem(e, "dataset", dataset.id)}
    >
      <LeadNumber label="Compounds" value={dataset.row_count.toLocaleString("en-US")} />
    </ItemCard>
  );
}

function DatasetGrid({ folderId }: { folderId: string | undefined }) {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Dataset[]>([]);
  const { data, isLoading, isError, error } = useDatasets(cursor, undefined, { folderId });
  const folders = useFolders("dataset").data;

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <>
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-40 w-full" />
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

      {data && items.length === 0 && folderId && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <DatasetsIcon className="mx-auto mb-3 size-6 text-icon-datasets opacity-60" />
          <p className="text-sm font-medium">
            This folder is empty. Drag a dataset here or use Move to folder.
          </p>
        </div>
      )}

      {data && items.length === 0 && !folderId && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <DatasetsIcon className="mx-auto mb-3 size-6 text-icon-datasets opacity-60" />
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
        <div className="grid items-stretch gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              draggable={folders?.can_edit ?? false}
            />
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
    </>
  );
}

export function DatasetList() {
  const { params, set } = useUrlParams();
  const folderId = params.get("folder") ?? undefined;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 p-2">
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

      <div className="flex flex-col gap-4 md:grid md:grid-cols-[13rem_minmax(0,1fr)] md:items-start md:gap-8">
        <FolderRail kind="dataset" activeId={folderId} onSelect={(id) => set({ folder: id })} />

        <div className="min-w-0 space-y-4">
          <DatasetGrid key={folderId ?? "all"} folderId={folderId} />
        </div>
      </div>
    </div>
  );
}
