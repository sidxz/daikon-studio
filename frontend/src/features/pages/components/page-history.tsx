"use client";

import { useQueryClient } from "@tanstack/react-query";
import { format, formatDistanceToNow } from "date-fns";
import { useState } from "react";
import { toast } from "sonner";

import { pagesKeys } from "@/features/pages/lib/constants";
import {
  currentVersion,
  usePageContent,
  useRevisePage,
  useRevisions,
} from "@/features/pages/lib/hooks";
import { displayName, useMemberIndex } from "@/features/pages/lib/members";
import type { PMDoc, PageView as PageViewData, RevisionView } from "@/features/pages/lib/types";
import { InitialsAvatar } from "@/shared/components/initials-avatar";
import { Button } from "@/shared/components/ui/button";
import { ConfirmDialog } from "@/shared/components/ui/confirm-dialog";
import { cn } from "@/shared/lib/utils";

import { PageView } from "./page-view";
import { VersionCompare } from "./version-compare";

/** Full-width history surface: version rail on the right, selected version
 *  rendered read-only in the main area. Restore appends the old body as a NEW
 *  revision (history is never rewritten). */
export function PageHistory({ page, onExit }: { page: PageViewData; onExit: () => void }) {
  const qc = useQueryClient();
  const { byId } = useMemberIndex();
  const { data: revisions } = useRevisions(page.id); // ascending revision_no
  const [selectedNo, setSelectedNo] = useState<number | null>(null); // null = current
  const [confirmRestore, setConfirmRestore] = useState(false);
  const [compare, setCompare] = useState(false);
  const revise = useRevisePage(page.id);

  const revs = revisions ?? [];
  const current = revs[revs.length - 1] ?? null;
  const selected =
    selectedNo === null ? current : (revs.find((r) => r.revision_no === selectedNo) ?? current);
  const isCurrent = selected?.revision_no === current?.revision_no;
  const { data: content } = usePageContent(page.id, selected?.sha256 ?? null);

  const ordinal = (r: RevisionView) => revs.findIndex((x) => x.revision_no === r.revision_no) + 1;
  const authorName = (id: string) => {
    const m = byId.get(id);
    return m ? displayName(m) : "Unknown";
  };

  const restore = () => {
    revise.mutate(
      { body: content as PMDoc, expectedVersion: currentVersion(qc, page.id) },
      {
        onSuccess: async () => {
          await qc.invalidateQueries({ queryKey: pagesKeys.revisions(page.id) });
          toast.success(`Restored Version ${selected ? ordinal(selected) : ""}`.trim());
          onExit();
        },
        onError: () => toast.error("Could not restore — the page changed underneath you"),
      },
    );
  };

  // Distinguish "still loading" from "resolved, genuinely empty" — collapsing
  // both into `revs.length === 0` flashed the empty state on every entry into
  // history mode (a within-surface layout jump once the query resolved).
  if (revisions === undefined) {
    return <div className="p-6 text-muted-foreground">Loading…</div>;
  }
  if (revs.length === 0) {
    return (
      <div className="p-6 text-sm text-muted-foreground">
        No saved versions yet — save the page once to start its history.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 gap-4 overflow-hidden">
      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col",
          compare ? "overflow-hidden" : "overflow-y-auto",
        )}
      >
        {compare ? (
          <VersionCompare
            pageId={page.id}
            revisions={revs}
            initialLeftNo={
              !isCurrent && selected
                ? selected.revision_no
                : (revs[revs.length - 2] ?? revs[0]).revision_no
            }
            initialRightNo={current!.revision_no}
          />
        ) : (
          <>
            {!isCurrent && selected && !page.archived && (
              <div className="mx-4 mt-3 flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/50 px-3 py-2 text-sm">
                <span>
                  Viewing Version {ordinal(selected)} — current is Version{" "}
                  {current ? ordinal(current) : "?"}
                </span>
                <Button size="sm" onClick={() => setConfirmRestore(true)} disabled={!content}>
                  Restore this version
                </Button>
              </div>
            )}
            {content !== undefined ? (
              <PageView key={selected?.sha256 ?? "empty"} content={(content as PMDoc) ?? null} />
            ) : (
              <div className="p-6 text-muted-foreground">Loading…</div>
            )}
          </>
        )}
      </div>
      <aside className="w-72 shrink-0 overflow-y-auto border-l border-border py-3 pl-4 pr-1">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-medium text-muted-foreground">Versions</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCompare((c) => !c)}
            disabled={revs.length < 2}
          >
            {compare ? "Back to preview" : "Compare"}
          </Button>
        </div>
        <ol>
          {[...revs].reverse().map((r) => (
            <li key={r.revision_no}>
              <button
                type="button"
                onClick={() => setSelectedNo(r.revision_no)}
                disabled={compare}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted",
                  selected?.revision_no === r.revision_no && "bg-muted",
                  // Compare mode ignores rail selection — the pickers are the only
                  // live control there, so mute the rail to avoid an inert control
                  // that looks responsive but does nothing.
                  compare && "cursor-default opacity-50 hover:bg-transparent",
                )}
              >
                <InitialsAvatar name={authorName(r.author_id)} id={r.author_id} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">
                    Version {ordinal(r)}
                    {r.revision_no === current?.revision_no && (
                      <span className="text-muted-foreground"> · current</span>
                    )}
                  </span>
                  <span className="block truncate text-xs text-muted-foreground">
                    {authorName(r.author_id)} ·{" "}
                    {formatDistanceToNow(new Date(r.created_at), { addSuffix: true })}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>
        <p className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">
          Created{page.created_at ? ` ${format(new Date(page.created_at), "PP")}` : ""} by{" "}
          {authorName(page.author_id)}
        </p>
      </aside>
      <ConfirmDialog
        open={confirmRestore}
        onOpenChange={setConfirmRestore}
        title={`Restore Version ${selected ? ordinal(selected) : ""}?`}
        description="The current version stays in the history — restoring adds this older content as a new version on top."
        confirmLabel="Restore"
        destructive={false}
        onConfirm={() => {
          setConfirmRestore(false);
          restore();
        }}
      />
    </div>
  );
}
