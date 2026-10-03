"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useProtocols } from "../hooks/use-protocols";
import type { Protocol } from "../types";

export function ProtocolList() {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Protocol[]>([]);
  const { data, isLoading, isError } = useProtocols(cursor);

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Protocols</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A trained model, scored against a fingerprint baseline. Drafts are yours alone;
            publishing locks one and makes it runnable by the whole workspace.
          </p>
        </div>
        <Button asChild>
          <Link href="/protocols/new">
            <Plus className="size-4" />
            Train a protocol
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
          <p className="text-sm font-medium text-destructive">Could not load protocols</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No protocols yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Freeze a dataset first, then train a model on it.
          </p>
        </div>
      )}

      {items.length > 0 && (
        <div className="grid items-stretch gap-3 sm:grid-cols-2">
          {items.map((protocol) => (
            <Link
              key={protocol.id}
              href={`/protocols/${protocol.id}`}
              className="flex h-full flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-muted/40"
            >
              <div className="flex items-start justify-between gap-3">
                <span className="font-medium">{protocol.name}</span>
                <Badge
                  variant={protocol.status === "draft" ? "outline" : "default"}
                  className="shrink-0 font-normal"
                >
                  {protocol.status === "draft" ? "Draft" : "Published"}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span className="font-mono">{protocol.engine_id}</span>
                <span>v{protocol.protocol_version}</span>
                <span>{new Date(protocol.created_at).toLocaleDateString()}</span>
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
