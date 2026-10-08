"use client";

import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  Archive,
  History as HistoryIcon,
  LockKeyhole,
  MoreHorizontal,
  Pencil,
  PencilLine,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { pagesApi } from "@/features/pages/lib/api";
import { pagesKeys } from "@/features/pages/lib/constants";
import {
  currentVersion,
  useArchivePage,
  useDeletePage,
  usePage,
  usePageContent,
  useRevisePage,
} from "@/features/pages/lib/hooks";
import { displayName, useMemberIndex } from "@/features/pages/lib/members";
import { pageSettings } from "@/features/pages/lib/page-settings";
import type { PMDoc, PageOwnerKind } from "@/features/pages/lib/types";
import { PageHeader } from "@/shared/components/page-header";
import { Button } from "@/shared/components/ui/button";
import { ConfirmDialog } from "@/shared/components/ui/confirm-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";

import { ConflictDialog } from "./conflict-dialog";
import { PageEditor } from "./page-editor";
import { PageHistory } from "./page-history";
import { PageView } from "./page-view";
import { RenamePageDialog } from "./rename-page-dialog";
import { usePageSave } from "./use-page-save";

const OWNER_LABEL: Record<PageOwnerKind, string> = {
  dataset: "Datasets",
  protocol: "Protocols",
  run: "Runs",
};

/** Where a page lives: its parent's detail view, opened on the notebook tab. */
export function ownerHref(kind: string, ownerId: string, pageId?: string): string {
  const page = pageId ? `&page=${pageId}` : "";
  return `/${kind}s/${ownerId}?tab=notebook${page}`;
}

export function PageContainer({ id, embedded = false }: { id: string; embedded?: boolean }) {
  const qc = useQueryClient();
  const { data: page, error } = usePage(id);
  const hasHead = !!page?.head_sha256;
  const { data: content, error: contentError } = usePageContent(id, page?.head_sha256 ?? null);
  const { byId } = useMemberIndex();

  const router = useRouter();
  const [mode, setMode] = useState<"view" | "edit" | "history">("view");
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [confirmUnlock, setConfirmUnlock] = useState(false);
  const [confirmArchive, setConfirmArchive] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const revisePage = useRevisePage(id);
  const archivePage = useArchivePage(id);
  const deletePage = useDeletePage();
  const locked = pageSettings((content as PMDoc) ?? null).locked;

  // One-shot handoff from the create sheet (?edit=1&layout=…&textSize=…),
  // captured at mount so stripping the params below can't blank it mid-flight.
  const search = useSearchParams();
  const [boot] = useState(() => ({
    edit: search.get("edit") === "1",
    layout: search.get("layout"),
    textSize: search.get("textSize"),
  }));
  // Settings picked at creation ride the fresh editor doc's attributes and are
  // persisted by the first ordinary save; only used when the page has no head.
  const [seed] = useState<PMDoc | null>(() =>
    boot.layout || boot.textSize
      ? {
          type: "doc",
          attrs: {
            ...(boot.layout ? { layout: boot.layout } : {}),
            ...(boot.textSize ? { textSize: boot.textSize } : {}),
          },
          content: [{ type: "paragraph" }],
        }
      : null,
  );

  // Land straight in the editor after "Create page". Render-phase state
  // adjustment (the set-state-in-effect lint bans the effect form), deferred
  // until content is loaded (edit mode must never mount TipTap on an undefined
  // doc) and gated the same way as the Edit button.
  const [booted, setBooted] = useState(false);
  if (boot.edit && !embedded && !booted && page && !(hasHead && content === undefined)) {
    setBooted(true);
    if (page.can_edit !== false && !locked && !page.archived) setMode("edit");
  }
  useEffect(() => {
    // Strip the one-shot params so a reload or a shared URL doesn't re-open
    // the editor.
    if (booted) window.history.replaceState(null, "", `/pages/${id}`);
  }, [booted, id]);

  // The full-page route names the trail; embedded, the host view already does.
  const kind = page?.owner_kind as PageOwnerKind | undefined;
  useBreadcrumbTrail(
    !embedded && page && kind
      ? [
          { label: OWNER_LABEL[kind], href: `/${kind}s` },
          { label: "Notebook", href: ownerHref(kind, page.owner_id, id) },
          { label: page.title },
        ]
      : null,
  );

  const { onChange, saveNow, discard, dirty, status, conflict, resolveConflict } = usePageSave({
    save: async (doc: PMDoc) => {
      await pagesApi.revise(id, doc, currentVersion(qc, id));
      await qc.invalidateQueries({ queryKey: pagesKeys.detail(id) }); // advances version for the next save
    },
  });

  // Unlock = a normal revision with locked:false — the audit trail is the
  // history rail itself.
  const unlock = () => {
    const doc = content as PMDoc;
    revisePage.mutate(
      {
        body: { ...doc, attrs: { ...(doc.attrs ?? {}), locked: false } },
        expectedVersion: currentVersion(qc, id),
      },
      {
        onSuccess: () => void qc.invalidateQueries({ queryKey: pagesKeys.revisions(id) }),
        onError: () => toast.error("Could not unlock — reload and try again"),
      },
    );
  };

  // Cmd/Ctrl+S accelerator; the Save button is the discoverable path.
  useEffect(() => {
    if (mode !== "edit") return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        // Mirror the Save button's disabled condition: a second Cmd+S while a
        // save is in flight would race the same cached expected_version and
        // surface a spurious 409 ConflictDialog on the user's own edits.
        if (!dirty || status === "saving") return;
        void saveNow();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, saveNow, dirty, status]);

  if (error || contentError) {
    return (
      <div className="p-6 text-sm text-destructive">
        Couldn’t load this page.{" "}
        <Link href="/" className="underline">
          Back to the studio
        </Link>
      </div>
    );
  }

  // usePage/usePageContent are plain useQuery — the route's <Suspense> only covers
  // the initial server-rendered shell, not these client fetches, so we render our
  // own loading state. Also wait for content when a head exists: PageEditor's
  // `initial` only applies at mount (TipTap doesn't react to prop changes), so
  // mounting it before content arrives would freeze on an empty doc.
  // Edit mode is exempt: it's only enterable from view mode, where content is
  // already loaded, and the fresh-sha refetch this component triggers after
  // every save must not unmount the live editor (that would drop the cursor
  // and undo stack, and silently lose any keystrokes typed during the round-trip).
  if (!page || (hasHead && content === undefined && mode !== "edit")) {
    return <div className="p-6 text-muted-foreground">Loading…</div>;
  }

  const statusText =
    status === "saving"
      ? "Saving…"
      : status === "error"
        ? "Couldn’t save — try again"
        : dirty
          ? "Unsaved changes"
          : status === "saved"
            ? "Saved"
            : "";

  // Falls back to the page creator for legacy revisions that predate author
  // stamping, so the caption always carries a name once members are loaded.
  const editor = byId.get(page.last_edited_by ?? page.author_id);
  const edited = `Edited ${formatDistanceToNow(new Date(page.updated_at), { addSuffix: true })}${
    editor ? ` by ${displayName(editor)}` : ""
  }`;

  const exitEdit = () => setMode("view");
  // Nothing is ever written implicitly, so leaving with unsaved edits has to be
  // an explicit, confirmed choice rather than a silent flush.
  const onDone = () => (dirty ? setConfirmDiscard(true) : exitEdit());

  return (
    <div className="flex h-full min-h-0 flex-col gap-1">
      {embedded ? (
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 pb-2">
          <div className="min-w-0 space-y-0.5">
            <h2 className="truncate text-lg font-medium">{page.title}</h2>
            {mode === "history" && <p className="text-xs text-muted-foreground">History</p>}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {mode === "view" ? (
              <>
                <span className="text-xs text-muted-foreground">{edited}</span>
                {embedded ? (
                  // Mounted as a section tab: History + Edit live in the ⋯ menu, and Edit
                  // opens the full page editor (route) rather than editing inside the mount.
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="icon" aria-label="Page actions">
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {!page.archived &&
                        page.can_edit !== false &&
                        (locked ? (
                          <DropdownMenuItem onSelect={() => setConfirmUnlock(true)}>
                            <LockKeyhole />
                            Unlock
                          </DropdownMenuItem>
                        ) : (
                          <DropdownMenuItem onSelect={() => router.push(`/pages/${id}?edit=1`)}>
                            <Pencil />
                            Edit
                          </DropdownMenuItem>
                        ))}
                      {!page.archived && page.can_edit !== false && (
                        <DropdownMenuItem onSelect={() => setRenaming(true)}>
                          <PencilLine />
                          Rename…
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuItem onSelect={() => setMode("history")}>
                        <HistoryIcon />
                        History
                      </DropdownMenuItem>
                      {!page.archived && page.can_archive !== false && (
                        <DropdownMenuItem onSelect={() => setConfirmArchive(true)}>
                          <Archive />
                          Archive page…
                        </DropdownMenuItem>
                      )}
                      {!page.archived && page.can_delete !== false && (
                        <DropdownMenuItem
                          variant="destructive"
                          onSelect={() => setConfirmDelete(true)}
                        >
                          <Trash2 />
                          Delete page…
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                ) : (
                  <>
                    <Button variant="outline" onClick={() => setMode("history")}>
                      <HistoryIcon />
                      History
                    </Button>
                    {!page.archived &&
                      page.can_edit !== false &&
                      (locked ? (
                        <Button variant="outline" onClick={() => setConfirmUnlock(true)}>
                          <LockKeyhole />
                          Unlock
                        </Button>
                      ) : (
                        <Button onClick={() => setMode("edit")}>
                          <Pencil />
                          Edit
                        </Button>
                      ))}
                    {/* Archived pages carry these actions on the banner instead. */}
                    {!page.archived && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" size="icon" aria-label="Page actions">
                            <MoreHorizontal />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          {page.can_edit !== false && (
                            <DropdownMenuItem onSelect={() => setRenaming(true)}>
                              <PencilLine />
                              Rename…
                            </DropdownMenuItem>
                          )}
                          {page.can_archive !== false && (
                            <DropdownMenuItem onSelect={() => setConfirmArchive(true)}>
                              <Archive />
                              Archive page…
                            </DropdownMenuItem>
                          )}
                          {page.can_delete !== false && (
                            <DropdownMenuItem
                              variant="destructive"
                              onSelect={() => setConfirmDelete(true)}
                            >
                              <Trash2 />
                              Delete page…
                            </DropdownMenuItem>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </>
                )}
              </>
            ) : mode === "history" ? (
              <Button variant="outline" onClick={() => setMode("view")}>
                Done
              </Button>
            ) : (
              <>
                <span aria-live="polite" className="text-sm text-muted-foreground">
                  {statusText}
                </span>
                <Button onClick={() => void saveNow()} disabled={!dirty || status === "saving"}>
                  Save
                </Button>
                <Button variant="outline" onClick={onDone}>
                  Done
                </Button>
              </>
            )}
          </div>
        </div>
      ) : (
        <PageHeader
          title={page.title}
          description={
            kind ? (
              <Link
                className="underline-offset-4 hover:underline"
                href={ownerHref(kind, page.owner_id, id)}
              >
                Back to the {kind}&apos;s notebook
              </Link>
            ) : undefined
          }
          action={
            mode === "view" ? (
              <>
                <span className="text-xs text-muted-foreground">{edited}</span>
                {embedded ? (
                  // Mounted as a section tab: History + Edit live in the ⋯ menu, and Edit
                  // opens the full page editor (route) rather than editing inside the mount.
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="icon" aria-label="Page actions">
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {!page.archived &&
                        page.can_edit !== false &&
                        (locked ? (
                          <DropdownMenuItem onSelect={() => setConfirmUnlock(true)}>
                            <LockKeyhole />
                            Unlock
                          </DropdownMenuItem>
                        ) : (
                          <DropdownMenuItem onSelect={() => router.push(`/pages/${id}?edit=1`)}>
                            <Pencil />
                            Edit
                          </DropdownMenuItem>
                        ))}
                      {!page.archived && page.can_edit !== false && (
                        <DropdownMenuItem onSelect={() => setRenaming(true)}>
                          <PencilLine />
                          Rename…
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuItem onSelect={() => setMode("history")}>
                        <HistoryIcon />
                        History
                      </DropdownMenuItem>
                      {!page.archived && page.can_archive !== false && (
                        <DropdownMenuItem onSelect={() => setConfirmArchive(true)}>
                          <Archive />
                          Archive page…
                        </DropdownMenuItem>
                      )}
                      {!page.archived && page.can_delete !== false && (
                        <DropdownMenuItem
                          variant="destructive"
                          onSelect={() => setConfirmDelete(true)}
                        >
                          <Trash2 />
                          Delete page…
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                ) : (
                  <>
                    <Button variant="outline" onClick={() => setMode("history")}>
                      <HistoryIcon />
                      History
                    </Button>
                    {!page.archived &&
                      page.can_edit !== false &&
                      (locked ? (
                        <Button variant="outline" onClick={() => setConfirmUnlock(true)}>
                          <LockKeyhole />
                          Unlock
                        </Button>
                      ) : (
                        <Button onClick={() => setMode("edit")}>
                          <Pencil />
                          Edit
                        </Button>
                      ))}
                    {/* Archived pages carry these actions on the banner instead. */}
                    {!page.archived && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" size="icon" aria-label="Page actions">
                            <MoreHorizontal />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          {page.can_edit !== false && (
                            <DropdownMenuItem onSelect={() => setRenaming(true)}>
                              <PencilLine />
                              Rename…
                            </DropdownMenuItem>
                          )}
                          {page.can_archive !== false && (
                            <DropdownMenuItem onSelect={() => setConfirmArchive(true)}>
                              <Archive />
                              Archive page…
                            </DropdownMenuItem>
                          )}
                          {page.can_delete !== false && (
                            <DropdownMenuItem
                              variant="destructive"
                              onSelect={() => setConfirmDelete(true)}
                            >
                              <Trash2 />
                              Delete page…
                            </DropdownMenuItem>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </>
                )}
              </>
            ) : mode === "history" ? (
              <Button variant="outline" onClick={() => setMode("view")}>
                Done
              </Button>
            ) : (
              <>
                <span aria-live="polite" className="text-sm text-muted-foreground">
                  {statusText}
                </span>
                <Button onClick={() => void saveNow()} disabled={!dirty || status === "saving"}>
                  Save
                </Button>
                <Button variant="outline" onClick={onDone}>
                  Done
                </Button>
              </>
            )
          }
        />
      )}
      {page.archived && mode !== "history" && (
        <div className="flex flex-wrap items-center gap-3 border-b border-border bg-muted/50 px-4 py-2 text-sm">
          <Archive className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          <span>This page is archived — it’s read-only and listed under archived pages.</span>
          {page.can_archive !== false && (
            <Button
              size="sm"
              variant="outline"
              disabled={archivePage.isPending}
              onClick={() =>
                archivePage.mutate(true, {
                  onSuccess: () => toast.success("Page restored"),
                })
              }
            >
              Restore page
            </Button>
          )}
          {page.can_delete !== false && (
            <Button
              size="sm"
              variant="outline"
              className="text-destructive"
              disabled={deletePage.isPending}
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2 />
              Delete page…
            </Button>
          )}
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        {mode === "edit" ? (
          // Constant key: saves invalidate the detail query and advance
          // head_sha256, and a sha-derived key would remount TipTap mid-typing
          // (losing cursor and undo stack) on every successful save.
          <PageEditor
            key="edit"
            initial={(content as PMDoc) ?? seed}
            onChange={onChange}
            pageId={id}
          />
        ) : mode === "history" ? (
          <PageHistory page={page} onExit={() => setMode("view")} />
        ) : (
          // Sha-keyed remount is how view mode picks up new content — TipTap
          // applies `initial` only at mount, so returning from an edit session
          // must force a fresh mount to show what was just saved.
          <PageView key={page.head_sha256 ?? "empty"} content={(content as PMDoc) ?? null} />
        )}
      </div>
      <ConfirmDialog
        open={confirmDiscard}
        onOpenChange={setConfirmDiscard}
        title="Discard unsaved changes?"
        description="This page has edits that haven’t been saved. Use Save to keep them."
        confirmLabel="Discard changes"
        onConfirm={() => {
          discard();
          setConfirmDiscard(false);
          exitEdit();
        }}
      />
      <ConflictDialog
        open={conflict}
        // A literal reload: the mounted editor won't pick up a swapped `initial` prop
        // (TipTap applies `content` only at mount), so an in-place cache invalidate
        // can't actually refresh what's on screen. The dialog copy says "reload" for
        // exactly this reason — ponytail: reload-on-conflict, not auto-merge.
        onReload={() => {
          // The user already chose to reload; discard first so use-page-save's
          // beforeunload guard (still armed while dirty) doesn't throw a native
          // "leave site?" prompt on top — cancelling that would strand them here.
          discard();
          window.location.reload();
        }}
        onDismiss={resolveConflict}
      />
      <ConfirmDialog
        open={confirmArchive}
        onOpenChange={setConfirmArchive}
        title="Archive this page?"
        description="Archived pages become read-only and move to the archived list. You can restore it at any time."
        confirmLabel="Archive page"
        destructive={false}
        pending={archivePage.isPending}
        onConfirm={() =>
          archivePage.mutate(false, {
            onSuccess: () => {
              setConfirmArchive(false);
              toast.success("Page archived");
            },
          })
        }
      />
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete this page?"
        description="The page and its whole version history are deleted for everyone. There is no undo."
        confirmLabel="Delete page"
        pending={deletePage.isPending}
        onConfirm={() =>
          deletePage.mutate(id, {
            onSuccess: () => {
              toast.success("Page deleted");
              if (!embedded && kind) router.push(ownerHref(kind, page.owner_id));
            },
          })
        }
      />
      <RenamePageDialog page={page} open={renaming} onOpenChange={setRenaming} />
      <ConfirmDialog
        open={confirmUnlock}
        onOpenChange={setConfirmUnlock}
        title="Unlock this page?"
        description="The page was locked to prevent accidental edits. Unlocking is recorded as a new version in its history."
        confirmLabel="Unlock"
        destructive={false}
        onConfirm={() => {
          setConfirmUnlock(false);
          unlock();
        }}
      />
    </div>
  );
}
