"use client";

import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import type { FolderKind } from "@/shared/lib/api/model";
import { Check, FolderInput } from "lucide-react";
import { useState } from "react";
import { useCreateFolder, useFileItem, useFolders } from "../hooks/use-folders";
import { FolderNameDialog } from "./folder-name-dialog";

/** The keyboard path to filing an item; dragging is the pointer one. Editors only. */
export function MoveToFolderMenu({
  kind,
  itemId,
  currentFolderId,
}: {
  kind: FolderKind;
  itemId: string;
  currentFolderId: string | null;
}) {
  const { data } = useFolders(kind);
  const file = useFileItem(kind);
  const create = useCreateFolder(kind);
  const [naming, setNaming] = useState(false);

  if (!data?.can_edit) return null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-7 shrink-0"
            aria-label="Move to folder"
          >
            <FolderInput className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {data.items.map((folder) => (
            <DropdownMenuItem
              key={folder.id}
              onSelect={() => file.mutate({ itemId, folderId: folder.id })}
            >
              {folder.name}
              {folder.id === currentFolderId && <Check className="ml-auto size-4" />}
            </DropdownMenuItem>
          ))}
          {data.items.length > 0 && <DropdownMenuSeparator />}
          <DropdownMenuItem onSelect={() => file.mutate({ itemId, folderId: null })}>
            No folder
            {currentFolderId === null && <Check className="ml-auto size-4" />}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setNaming(true)}>New folder…</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {naming && (
        <FolderNameDialog
          open
          onOpenChange={(open) => !open && setNaming(false)}
          title="New folder"
          onSubmit={async (name) => {
            const folder = await create.mutateAsync(name);
            await file.mutateAsync({ itemId, folderId: folder.id });
          }}
        />
      )}
    </>
  );
}
