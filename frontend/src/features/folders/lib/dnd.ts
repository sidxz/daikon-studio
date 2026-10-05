import type { DragEvent } from "react";

export const ITEM_MIME = "application/x-studio-item";

export type DragKind = "dataset" | "protocol";

/**
 * Browsers hide a drag's data until the drop, but not its types. The kind is also
 * written as a type of its own, so a tile can tell during dragover whether to accept.
 */
export function setDragItem(e: DragEvent, kind: DragKind, id: string): void {
  e.dataTransfer.setData(ITEM_MIME, JSON.stringify({ kind, id }));
  e.dataTransfer.setData(`${ITEM_MIME}.${kind}`, "");
  e.dataTransfer.effectAllowed = "move";
}

/** Takes a React or a native drag event: the rail also watches drags at the document. */
export function canDropItem(e: { dataTransfer: DataTransfer | null }, kind: DragKind): boolean {
  return Array.from(e.dataTransfer?.types ?? []).includes(`${ITEM_MIME}.${kind}`);
}

export function readDragItem(e: DragEvent): { kind: DragKind; id: string } | null {
  try {
    const item = JSON.parse(e.dataTransfer.getData(ITEM_MIME));
    return (item?.kind === "dataset" || item?.kind === "protocol") && typeof item.id === "string"
      ? item
      : null;
  } catch {
    return null;
  }
}
