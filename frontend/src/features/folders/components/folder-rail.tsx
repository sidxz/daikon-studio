"use client";

import { DatasetsIcon, ProtocolsIcon } from "@/shared/components/icons/nav-icons";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/shared/components/ui/alert-dialog";
import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import type { FolderKind, FolderResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { Folder, FolderOpen, MoreHorizontal, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import {
  useCreateFolder,
  useDeleteFolder,
  useFileItem,
  useFolders,
  useRenameFolder,
} from "../hooks/use-folders";
import { canDropItem, readDragItem } from "../lib/dnd";
import { FolderNameDialog } from "./folder-name-dialog";

// Written out in full so Tailwind sees every class: one nav hue per page.
const HUE = {
  dataset: {
    active: "border-transparent bg-icon-datasets/10 text-icon-datasets",
    over: "bg-icon-datasets/20",
    dashed: "outline-dashed outline-1 -outline-offset-1 outline-icon-datasets/60",
  },
  protocol: {
    active: "border-transparent bg-icon-protocols/10 text-icon-protocols",
    over: "bg-icon-protocols/20",
    dashed: "outline-dashed outline-1 -outline-offset-1 outline-icon-protocols/60",
  },
} as const;

const COPY = {
  dataset: { all: "All datasets", plural: "datasets", Icon: DatasetsIcon },
  protocol: { all: "All protocols", plural: "protocols", Icon: ProtocolsIcon },
} as const;

// A chip in the phone-width row, a borderless row in the rail from md up.
const row =
  "group/row flex shrink-0 items-center rounded-md border text-sm transition-colors md:border-transparent";
const idle = "border-border text-muted-foreground hover:text-foreground";
const select =
  "flex min-w-0 flex-1 items-center gap-2 whitespace-nowrap rounded-md px-3 py-1.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring md:px-2";

/** The folders of one kind, as a filter. Editors also file items by dropping cards on a folder. */
export function FolderRail({
  kind,
  activeId,
  onSelect,
}: {
  kind: FolderKind;
  activeId: string | undefined;
  onSelect: (id: string | undefined) => void;
}) {
  const { data } = useFolders(kind);
  const create = useCreateFolder(kind);
  const rename = useRenameFolder(kind);
  const remove = useDeleteFolder(kind);
  const file = useFileItem(kind);
  const [naming, setNaming] = useState<FolderResponse | "new" | null>(null);
  const [deleting, setDeleting] = useState<FolderResponse | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const canEdit = data?.can_edit ?? false;
  const folders = data?.items ?? [];
  const hue = HUE[kind];
  const { all, plural, Icon } = COPY[kind];

  // A card's drag starts outside the rail, so watch the whole page for one of this kind.
  useEffect(() => {
    if (!canEdit) return;
    const start = (e: DragEvent) => setDragging(canDropItem(e, kind));
    const end = () => {
      setDragging(false);
      setDropTarget(null);
    };
    document.addEventListener("dragstart", start);
    document.addEventListener("dragend", end);
    document.addEventListener("drop", end);
    return () => {
      document.removeEventListener("dragstart", start);
      document.removeEventListener("dragend", end);
      document.removeEventListener("drop", end);
    };
  }, [canEdit, kind]);

  return (
    <nav aria-label="Folders" className="min-w-0 space-y-2 md:sticky md:top-4">
      <h2 className="hidden px-2 text-xs text-muted-foreground md:block">
        {dragging && folders.length > 0 ? "Drop on a folder to file it" : "Folders"}
      </h2>
      <ul className="flex gap-1.5 overflow-x-auto pb-1 md:flex-col md:gap-0.5 md:overflow-visible md:pb-0">
        <li className={cn(row, !activeId ? hue.active : idle)}>
          <button
            type="button"
            aria-pressed={!activeId}
            onClick={() => onSelect(undefined)}
            className={select}
          >
            <Icon className="size-4 shrink-0" />
            {all}
          </button>
        </li>

        {folders.map((folder) => {
          const active = folder.id === activeId;
          const FolderIcon = active ? FolderOpen : Folder;
          return (
            <li
              key={folder.id}
              data-testid={`folder-${folder.id}`}
              className={cn(
                row,
                active ? hue.active : idle,
                dragging && hue.dashed,
                dropTarget === folder.id && hue.over,
              )}
              onDragOver={(e) => {
                if (!canEdit || !canDropItem(e, kind)) return;
                e.preventDefault();
                setDropTarget(folder.id);
              }}
              onDragLeave={() => setDropTarget(null)}
              onDrop={(e) => {
                setDropTarget(null);
                const item = readDragItem(e);
                if (!canEdit || item?.kind !== kind) return;
                e.preventDefault();
                file.mutate({ itemId: item.id, folderId: folder.id });
              }}
            >
              <button
                type="button"
                aria-pressed={active}
                onClick={() => onSelect(folder.id)}
                className={select}
              >
                <FolderIcon className="size-4 shrink-0" />
                <span className="max-w-40 truncate md:max-w-none" title={folder.name}>
                  {folder.name}
                </span>
                <span className="ml-auto pl-1 text-xs tabular-nums text-muted-foreground">
                  {folder.item_count}
                </span>
              </button>
              {canEdit && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="mr-1 size-6 opacity-0 group-hover/row:opacity-100 group-focus-within/row:opacity-100 data-[state=open]:opacity-100 [@media(hover:none)]:opacity-100"
                      aria-label={`Folder actions for ${folder.name}`}
                    >
                      <MoreHorizontal className="size-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuItem onSelect={() => setNaming(folder)}>Rename</DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => setDeleting(folder)}>Delete</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </li>
          );
        })}

        {canEdit && (
          <li className={cn(row, "border-transparent text-muted-foreground hover:text-foreground")}>
            <button type="button" onClick={() => setNaming("new")} className={select}>
              <Plus className="size-4 shrink-0" />
              New folder
            </button>
          </li>
        )}
      </ul>

      {naming && (
        <FolderNameDialog
          open
          onOpenChange={(open) => !open && setNaming(null)}
          title={naming === "new" ? "New folder" : "Rename folder"}
          initialName={naming === "new" ? "" : naming.name}
          onSubmit={(name) =>
            naming === "new"
              ? create.mutateAsync(name)
              : rename.mutateAsync({ id: naming.id, name })
          }
        />
      )}

      <AlertDialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete folder</AlertDialogTitle>
            <AlertDialogDescription>
              Delete the folder "{deleting?.name}"? The {plural} in it stay in the workspace and
              move to No folder.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!deleting) return;
                if (deleting.id === activeId) onSelect(undefined);
                remove.mutate(deleting.id);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </nav>
  );
}
