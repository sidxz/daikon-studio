"use client";

import { type NodeViewProps, NodeViewWrapper } from "@tiptap/react";
import { Pencil, Trash2 } from "lucide-react";
import { type ReactNode, createContext, useContext } from "react";

import { Button } from "@/shared/components/ui/button";
import { cn } from "@/shared/lib/utils";

import type { EmbedKind } from "./slash-menu";

export type EmbedWidth = "full" | "half" | "fit";

/** Applied to the embed's <figure>; every renderer inside fills its container.
 *  Widths are presets rather than drag-resize handles — they stay meaningful
 *  when the same page is read on a narrower screen, and they're reachable from
 *  the keyboard. */
export const EMBED_WIDTH_CLASS: Record<EmbedWidth, string> = {
  full: "w-full",
  half: "w-full sm:w-1/2",
  // An explicit narrow column, NOT w-fit: the 3D viewer's canvas is
  // position:absolute, so it contributes no intrinsic width and a shrink-to-fit
  // container collapses to zero. A concrete width behaves the same for every
  // embed type. The stored attr value stays "fit" so existing nodes still read.
  fit: "w-full sm:w-80",
};

const WIDTH_OPTIONS: { value: EmbedWidth; label: string }[] = [
  { value: "full", label: "Full" },
  { value: "half", label: "Half" },
  { value: "fit", label: "Small" },
];

/** Re-opening an embed's insert dialog on an existing node. `onUpdate` is that
 *  node view's own updateAttributes, so the dialog writes back to *that* node
 *  and doesn't depend on where the selection sits when it confirms — opening a
 *  dialog moves DOM focus out of the editor, so selection is not a safe handle. */
export type EmbedEditRequest = {
  attrs: Record<string, unknown>;
  onUpdate: (attrs: Record<string, unknown>) => void;
};

/** `preset` names a CHART_PRESETS entry when the slash menu opened a named chart
 *  (volcano, ROC, …). It seeds a *new* node's defaults, so it is mutually
 *  exclusive with `editing`, which seeds from an existing one. */
export type OpenEmbedDialog = (
  kind: EmbedKind,
  editing?: EmbedEditRequest,
  preset?: string,
) => void;

// Node views render through TipTap portals, and those portals are created inside
// EditorContent — still under PageEditor in the React tree — so plain context
// reaches them. No need to thread a callback through every Node's options.
const EmbedDialogContext = createContext<OpenEmbedDialog>(() => {});

export const EmbedDialogProvider = EmbedDialogContext.Provider;
export const useEmbedDialog = () => useContext(EmbedDialogContext);

/**
 * The controls that appear on a selected embed: size presets, Edit (reopens the
 * insert dialog with current values), and Remove. Follows the floating
 * selection-toolbar convention used by Confluence and Notion — frequent layout
 * choices inline, everything else behind Edit.
 *
 * Clicking a 3D viewer's canvas still selects the node (ProseMirror gives it a
 * NodeSelection), but DOM focus lands on the canvas rather than the editor, so
 * keyboard Delete never arrives. That's why Remove is a visible button and not
 * left to the Delete key.
 */
export function EmbedToolbar({
  editor,
  selected,
  width,
  updateAttributes,
  deleteNode,
  onEdit,
}: Pick<NodeViewProps, "editor" | "selected" | "updateAttributes" | "deleteNode"> & {
  width: EmbedWidth;
  onEdit: () => void;
}) {
  if (!selected || !editor.isEditable) return null;

  return (
    <div
      role="toolbar"
      aria-label="Embed options"
      // contentEditable={false} keeps ProseMirror from treating these buttons as
      // document content; the overlay position means showing it never reflows
      // the page (no layout jump).
      contentEditable={false}
      className="absolute top-2 right-2 z-10 flex items-center gap-1 rounded-md border border-border bg-background p-1 shadow-md"
    >
      {WIDTH_OPTIONS.map((o) => (
        <Button
          key={o.value}
          type="button"
          size="xs"
          variant="ghost"
          aria-pressed={width === o.value}
          className={cn(width === o.value && "bg-muted text-foreground")}
          onClick={() => updateAttributes({ width: o.value })}
        >
          {o.label}
        </Button>
      ))}
      <span aria-hidden className="mx-0.5 h-4 w-px bg-border" />
      <Button type="button" size="xs" variant="ghost" onClick={onEdit}>
        <Pencil />
        Edit
      </Button>
      <Button type="button" size="xs" variant="ghost" onClick={deleteNode}>
        <Trash2 />
        Remove
      </Button>
    </div>
  );
}

/**
 * The shared shell every embed node view renders into: a width-constrained
 * <figure> with the selection toolbar overlaid, and Edit wired back to the
 * insert dialog that created this node kind. Each view supplies only its own
 * content (depiction/viewer/track plus captions) as children.
 */
export function EmbedFigure({
  kind,
  width,
  node,
  editor,
  selected,
  updateAttributes,
  deleteNode,
  children,
}: Pick<NodeViewProps, "node" | "editor" | "selected" | "updateAttributes" | "deleteNode"> & {
  kind: EmbedKind;
  width: EmbedWidth;
  children: ReactNode;
}) {
  const openDialog = useEmbedDialog();

  return (
    <NodeViewWrapper as="figure" className={cn("relative my-4", EMBED_WIDTH_CLASS[width])}>
      <EmbedToolbar
        editor={editor}
        selected={selected}
        width={width}
        updateAttributes={updateAttributes}
        deleteNode={deleteNode}
        onEdit={() => openDialog(kind, { attrs: node.attrs, onUpdate: updateAttributes })}
      />
      {children}
    </NodeViewWrapper>
  );
}
