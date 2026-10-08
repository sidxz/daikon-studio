"use client";

import { formatDistanceToNow } from "date-fns";
import { Archive, NotebookText, Plus } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { useCreatePage, useOwnerPages } from "@/features/pages/lib/hooks";
import type { PageOwnerKind } from "@/features/pages/lib/types";
import { Button } from "@/shared/components/ui/button";
import { EmptyState } from "@/shared/components/ui/empty-state";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { cn } from "@/shared/lib/utils";

import { PageContainer } from "./page-container";
import { PageTitleDialog } from "./page-title-dialog";

const NOUN: Record<PageOwnerKind, string> = {
  dataset: "dataset",
  protocol: "protocol",
  run: "run",
};

/**
 * A dataset's, protocol's or run's notebook: the pages mounted on it, listed on
 * the left, the chosen one read on the right. Writing happens on the page's own
 * route (`/pages/{id}`), which gives the editor the full width.
 */
export function NotebookPanel({
  kind,
  ownerId,
  canCreate = true,
}: {
  kind: PageOwnerKind;
  ownerId: string;
  canCreate?: boolean;
}) {
  const router = useRouter();
  const params = useSearchParams();
  const [showArchived, setShowArchived] = useState(false);
  const [creating, setCreating] = useState(false);
  const [picked, setPicked] = useState<string | null>(() => params.get("page"));
  const { data: pages, isLoading, error } = useOwnerPages(kind, ownerId, showArchived);
  const create = useCreatePage(kind, ownerId);

  const selected = pages?.find((p) => p.id === picked) ?? pages?.[0];

  const newPage = (title: string) =>
    create.mutate(title, {
      onSuccess: (page) => router.push(`/pages/${page.id}?edit=1`),
    });

  const createButton = canCreate && (
    <Button size="sm" onClick={() => setCreating(true)}>
      <Plus />
      New page
    </Button>
  );

  return (
    <div className="grid gap-6 md:grid-cols-[16rem_minmax(0,1fr)]">
      <aside className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-medium">{showArchived ? "Archived pages" : "Pages"}</h2>
          {!showArchived && createButton}
        </div>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : error ? (
          <p className="text-sm text-destructive">Couldn’t load the notebook.</p>
        ) : pages && pages.length > 0 ? (
          <ul className="flex flex-col gap-1">
            {pages.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  onClick={() => setPicked(p.id)}
                  aria-current={selected?.id === p.id ? "page" : undefined}
                  className={cn(
                    "w-full rounded-md px-3 py-2 text-left transition-colors hover:bg-muted",
                    selected?.id === p.id && "bg-muted",
                  )}
                >
                  <span className="block truncate text-sm font-medium">{p.title}</span>
                  <span className="block text-xs text-muted-foreground">
                    {`Edited ${formatDistanceToNow(new Date(p.updated_at), { addSuffix: true })}`}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">
            {showArchived ? "No archived pages." : "No pages yet."}
          </p>
        )}
        <button
          type="button"
          className="flex items-center gap-1.5 self-start text-xs text-muted-foreground underline-offset-4 hover:underline"
          onClick={() => {
            setShowArchived((v) => !v);
            setPicked(null);
          }}
        >
          <Archive className="size-3.5" aria-hidden />
          {showArchived ? "Back to pages" : "Show archived"}
        </button>
      </aside>

      <section className="min-w-0">
        {selected ? (
          <PageContainer key={selected.id} id={selected.id} embedded />
        ) : isLoading ? null : (
          <EmptyState
            icon={NotebookText}
            title={showArchived ? "Nothing archived" : `No notes on this ${NOUN[kind]} yet`}
            description={
              showArchived
                ? undefined
                : "Pages record what you did and why: conditions, observations, figures, structures."
            }
            action={showArchived ? undefined : createButton || undefined}
          />
        )}
      </section>

      <PageTitleDialog
        open={creating}
        onOpenChange={setCreating}
        title="New page"
        description={`A notebook page on this ${NOUN[kind]}. You can rename it later.`}
        confirmLabel="Create page"
        pending={create.isPending}
        onSubmit={newPage}
      />
    </div>
  );
}
