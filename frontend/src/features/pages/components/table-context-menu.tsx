"use client";

import { type Editor, useEditorState } from "@tiptap/react";
import {
  ArrowDownToLine,
  ArrowLeftToLine,
  ArrowRightToLine,
  ArrowUpToLine,
  Heading,
  Trash2,
} from "lucide-react";
import type { ReactNode } from "react";

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/shared/components/ui/context-menu";

/**
 * Word-style right-click editing for tables: row/column insert and delete live
 * here rather than in the main toolbar, because they only ever apply to the cell
 * under the pointer.
 *
 * Outside a table the trigger is disabled, which lets the browser's native menu
 * through — so spellcheck, copy/paste and look-up stay available in ordinary
 * prose instead of being replaced by an empty menu.
 */
export function TableContextMenu({ editor, children }: { editor: Editor; children: ReactNode }) {
  const inTable = useEditorState({
    editor,
    selector: ({ editor: e }) => e.isActive("table"),
  });

  const chain = () => editor.chain().focus();

  return (
    <ContextMenu>
      <ContextMenuTrigger disabled={!inTable} asChild>
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-52">
        <ContextMenuItem onSelect={() => chain().addRowBefore().run()}>
          <ArrowUpToLine />
          Insert row above
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => chain().addRowAfter().run()}>
          <ArrowDownToLine />
          Insert row below
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => chain().addColumnBefore().run()}>
          <ArrowLeftToLine />
          Insert column left
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => chain().addColumnAfter().run()}>
          <ArrowRightToLine />
          Insert column right
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => chain().toggleHeaderRow().run()}>
          <Heading />
          Toggle header row
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem variant="destructive" onSelect={() => chain().deleteRow().run()}>
          <Trash2 />
          Delete row
        </ContextMenuItem>
        <ContextMenuItem variant="destructive" onSelect={() => chain().deleteColumn().run()}>
          <Trash2 />
          Delete column
        </ContextMenuItem>
        <ContextMenuItem variant="destructive" onSelect={() => chain().deleteTable().run()}>
          <Trash2 />
          Delete table
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
