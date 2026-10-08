"use client";

import { FolderTabs, MoveToFolderMenu, setDragItem, useFolders } from "@/features/folders";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { DatasetsIcon } from "@/shared/components/icons/nav-icons";
import { ItemCard } from "@/shared/components/item-card";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Button } from "@/shared/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/shared/components/ui/sheet";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Eye, Plus, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import {
  type DatasetFilters as DatasetFilterValues,
  useDatasetCompounds,
  useDatasets,
} from "../hooks/use-datasets";
import { type Dataset, SPLIT_COPY, type SplitStrategy } from "../types";
import { DatasetFilters } from "./dataset-filters";

/** Fetch structures only when requested; opening a list never reads every snapshot. */
function DatasetSample({ dataset }: { dataset: Dataset }) {
  const sample = useDatasetCompounds(dataset.id, { offset: 0, limit: 3 });
  return (
    <section aria-label={`Structure preview for ${dataset.name}`} className="px-4 pb-5">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm">Structure sample</h3>
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          First {sample.data?.items.length ?? 3} compounds · measured values
        </span>
      </div>
      {sample.isLoading && <Skeleton className="h-28 w-full" />}
      {sample.isError && (
        <QueryError
          title="Could not load structures"
          retry={() => sample.refetch()}
          retrying={sample.isFetching}
        />
      )}
      {sample.data && (
        <div className="grid gap-6 sm:grid-cols-3">
          {sample.data.items.map((compound, index) => (
            <div
              key={`${index}-${compound.structure}`}
              className="flex min-w-0 items-center gap-3 sm:flex-col sm:items-start"
            >
              <StructureThumbnail smiles={compound.structure} size={92} className="shrink-0" />
              <div className="min-w-0 space-y-1">
                {compound.compound_id && (
                  <p className="truncate font-mono text-xs" title={compound.compound_id}>
                    {compound.compound_id}
                  </p>
                )}
                {dataset.targets.map((target) => (
                  <p key={target.column} className="text-xs">
                    <span className="mr-2 text-muted-foreground">{target.column}</span>
                    <ReadoutValue
                      value={compound.targets[target.column]}
                      unit={target.unit}
                      precision={target.kind === "binary" ? 0 : 3}
                      className="font-mono"
                    />
                  </p>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <Link
        href={`/datasets/${dataset.id}`}
        className="mt-5 inline-block text-xs text-primary underline underline-offset-4"
      >
        Open dataset →
      </Link>
    </section>
  );
}

export function DatasetCard({ dataset, draggable }: { dataset: Dataset; draggable: boolean }) {
  const creator = useMemberName()(dataset.created_by);
  const [preview, setPreview] = useState(false);
  const [target, ...others] = dataset.targets;
  return (
    <div className="h-full">
      <ItemCard
        compact
        href={`/datasets/${dataset.id}`}
        name={dataset.name}
        subtitle={
          target && (
            <span title={dataset.targets.map((t) => t.column).join(", ")}>
              Targets {target.column}
              {others.length > 0 && ` and ${others.length} more`}
            </span>
          )
        }
        action={
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Preview structures in ${dataset.name}`}
              aria-expanded={preview}
              aria-controls={`sample-${dataset.id}`}
              onClick={() => setPreview(!preview)}
            >
              <Eye />
            </Button>
            <MoveToFolderMenu
              kind="dataset"
              itemId={dataset.id}
              currentFolderId={dataset.folder_id}
            />
          </div>
        }
        footerStart={`${SPLIT_COPY[dataset.split.strategy].title} split`}
        creator={creator}
        creatorId={dataset.created_by}
        createdAt={dataset.created_at}
        draggable={draggable}
        onDragStart={(e) => setDragItem(e, "dataset", dataset.id)}
      >
        <div className="flex items-baseline gap-1.5">
          <span className="text-lg font-semibold tabular-nums">
            {dataset.row_count.toLocaleString("en-US")}
          </span>
          <span className="text-[11px] text-muted-foreground">compounds</span>
        </div>
      </ItemCard>
      <Sheet open={preview} onOpenChange={setPreview}>
        <SheetContent id={`sample-${dataset.id}`} className="w-full overflow-y-auto sm:max-w-xl">
          <SheetHeader>
            <SheetTitle className="font-sans">{dataset.name}</SheetTitle>
            <SheetDescription>Structures and measured targets from this dataset.</SheetDescription>
          </SheetHeader>
          {preview && <DatasetSample dataset={dataset} />}
        </SheetContent>
      </Sheet>
    </div>
  );
}

function DatasetGrid({ filters, onClear }: { filters: DatasetFilterValues; onClear: () => void }) {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Dataset[]>([]);
  const { data, isLoading, isError, refetch, isFetching } = useDatasets(cursor, undefined, filters);
  const folderId = filters.folderId;
  const filtered = Boolean(filters.q || filters.targetKind || filters.splitStrategy);
  const folders = useFolders("dataset").data;

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <>
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      )}

      {isError && (
        <QueryError title="Could not load datasets" retry={() => refetch()} retrying={isFetching} />
      )}

      {data && items.length === 0 && filtered && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <Search className="mx-auto mb-3 size-5 text-muted-foreground" aria-hidden />
          <p className="text-sm font-medium">No datasets match these filters.</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try another search or clear the filters.
          </p>
          <Button variant="outline" className="mt-4" onClick={onClear}>
            Clear filters
          </Button>
        </div>
      )}

      {data && items.length === 0 && !filtered && folderId && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <DatasetsIcon className="mx-auto mb-3 size-6 text-icon-datasets opacity-60" />
          <p className="text-sm font-medium">
            This folder is empty. Drag a dataset here or use Move to folder.
          </p>
        </div>
      )}

      {data && items.length === 0 && !filtered && !folderId && (
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
        <div className="grid items-stretch gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
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
  const q = params.get("q") ?? "";
  const kind = params.get("target_kind");
  // Validated against the copy table, which is keyed by the generated enum, so
  // a new strategy is filterable the day it ships instead of being dropped here.
  const split = params.get("split_strategy") as SplitStrategy | null;
  const filters: DatasetFilterValues = {
    folderId,
    q: q || undefined,
    targetKind: kind === "numeric" || kind === "binary" ? kind : undefined,
    splitStrategy: split != null && SPLIT_COPY[split] != null ? split : undefined,
  };
  const clear = () => set({ q: undefined, target_kind: undefined, split_strategy: undefined });

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Datasets"
        description="Structures, measured targets, and fixed training splits."
        action={
          <Button asChild>
            <Link href="/datasets/new">
              <Plus className="size-4" />
              New dataset
            </Link>
          </Button>
        }
      />

      <div className="min-w-0 space-y-4">
        <FolderTabs kind="dataset" activeId={folderId} onSelect={(id) => set({ folder: id })} />

        <DatasetFilters
          q={q}
          targetKind={filters.targetKind}
          splitStrategy={filters.splitStrategy}
          onChange={set}
        />

        <div className="min-w-0 space-y-4">
          <DatasetGrid key={JSON.stringify(filters)} filters={filters} onClear={clear} />
        </div>
      </div>
    </div>
  );
}
