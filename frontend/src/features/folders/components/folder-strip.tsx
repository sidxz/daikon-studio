"use client";

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
import { MoreHorizontal, Plus } from "lucide-react";
import { useState } from "react";
import {
  useCreateFolder,
  useDeleteFolder,
  useFileItem,
  useFolders,
  useRenameFolder,
} from "../hooks/use-folders";
import { canDropItem, readDragItem } from "../lib/dnd";
import { FolderNameDialog } from "./folder-name-dialog";

const tile =
  "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition-colors";

export function FolderStrip({
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

  const canEdit = data?.can_edit ?? false;
  const plural = kind === "dataset" ? "datasets" : "protocols";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        aria-pressed={!activeId}
        onClick={() => onSelect(undefined)}
        className={cn(tile, !activeId ? "border-foreground/40 bg-muted" : "hover:bg-muted/40")}
      >
        All
      </button>

      {data?.items.map((folder) => (
        <div
          key={folder.id}
          data-testid={`folder-${folder.id}`}
          className={cn(
            tile,
            "py-0 pr-1",
            folder.id === activeId ? "border-foreground/40 bg-muted" : "hover:bg-muted/40",
            dropTarget === folder.id && "border-primary bg-primary/10",
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
            aria-pressed={folder.id === activeId}
            onClick={() => onSelect(folder.id)}
            className="flex items-center gap-1.5 py-1.5"
          >
            {folder.name}
            <span className="text-xs text-muted-foreground">{folder.item_count}</span>
          </button>
          {canEdit && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-6"
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
        </div>
      ))}

      {canEdit && (
        <Button variant="ghost" size="sm" onClick={() => setNaming("new")}>
          <Plus className="size-4" />
          New folder
        </Button>
      )}

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
    </div>
  );
}
