"use client";

import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import Link from "next/link";
import { useState } from "react";
import { useCollections } from "../hooks/use-collections";
import type { Collection } from "../types";

export function CollectionList() {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Collection[]>([]);
  const { data, isLoading, isError } = useCollections(cursor);

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div>
        <h1 className="text-lg font-semibold">Collections</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          A saved triage decision — the compounds you were willing to order, frozen with the run
          they came from and marked as AI-predicted.
        </p>
      </div>

      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load collections</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No collections yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Run a published protocol, then pick the compounds worth pursuing from its results.
          </p>
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((collection) => (
            <Link
              key={collection.id}
              href={`/collections/${collection.id}`}
              className="flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
            >
              <span className="font-medium">{collection.name}</span>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>{collection.member_count.toLocaleString()} compounds</span>
                <span>{new Date(collection.created_at).toLocaleDateString()}</span>
              </div>
            </Link>
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
    </div>
  );
}
