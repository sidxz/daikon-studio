"use client";

import "katex/dist/katex.min.css";

import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import { Details, DetailsContent, DetailsSummary } from "@tiptap/extension-details";
import { FileHandler } from "@tiptap/extension-file-handler";
import Highlight from "@tiptap/extension-highlight";
import Image from "@tiptap/extension-image";
// katex is pinned to 0.17 in package.json: extension-mathematics@3.28.0
// peer-requires ^0.16.4 || ^0.17.0, and a bare `pnpm add katex` resolves 0.18.
import { Mathematics } from "@tiptap/extension-mathematics";
import Subscript from "@tiptap/extension-subscript";
import Superscript from "@tiptap/extension-superscript";
// @tiptap/extension-table@3.28.0 has no default export (only Table/TableRow/
// TableCell/TableHeader as named exports — the sub-packages re-export those with a
// default, but the parent package itself doesn't), unlike the brief's sketch.
import { Table } from "@tiptap/extension-table";
import TableCell from "@tiptap/extension-table-cell";
import TableHeader from "@tiptap/extension-table-header";
import TableRow from "@tiptap/extension-table-row";
import type { Node as PMNode } from "@tiptap/pm/model";
import { type Content, type Editor, EditorContent, useEditor, useEditorState } from "@tiptap/react";
import { common, createLowlight } from "lowlight";
import { type ComponentType, useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { uploadBlob } from "@/features/pages/lib/blobs";
import { sharedExtensions } from "@/features/pages/lib/editor/core";
import { searchEntities } from "@/features/pages/lib/entity-search";
import { searchMemberItems } from "@/features/pages/lib/members";
import { pageSettings, pageSettingsClasses } from "@/features/pages/lib/page-settings";
import type { DiffProp } from "@/features/pages/lib/pm-diff";
import type { PMDoc } from "@/features/pages/lib/types";
import { cn } from "@/shared/lib/utils";

import { InsertChartDialog } from "./dialogs/insert-chart-dialog";
import { InsertFigureDialog } from "./dialogs/insert-figure-dialog";
import { InsertMathDialog } from "./dialogs/insert-math-dialog";
import { InsertMoleculeDialog } from "./dialogs/insert-molecule-dialog";
import { InsertProteinDialog } from "./dialogs/insert-protein-dialog";
import { InsertReactionDialog } from "./dialogs/insert-reaction-dialog";
import { InsertSequenceDialog } from "./dialogs/insert-sequence-dialog";
import { InsertStructureGridDialog } from "./dialogs/insert-structure-grid-dialog";
import { diffHighlight } from "./diff-highlight";
import { EditorToolbar } from "./editor-toolbar";
import { EmbedDialogProvider, type EmbedEditRequest, type OpenEmbedDialog } from "./embed-toolbar";
import { Chart } from "./nodes/chart";
import { ChemStructure } from "./nodes/chem-structure";
import { EntityLink } from "./nodes/entity-link";
import { FigureImage } from "./nodes/figure-image";
import { PageDoc } from "./nodes/page-doc";
import { ProteinStructure } from "./nodes/protein-structure";
import { ReactionScheme } from "./nodes/reaction-scheme";
import { SequenceViewer } from "./nodes/sequence-viewer";
import { StructureGrid } from "./nodes/structure-grid";
import { type EmbedKind, createSlashMenu } from "./slash-menu";
import { TableContextMenu } from "./table-context-menu";

type EmbedDialogProps = {
  editor: Editor;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Absent when inserting; present when the EmbedToolbar reopened the dialog on
   *  an existing node, in which case confirming updates it instead of inserting. */
  editing?: EmbedEditRequest;
  /** A CHART_PRESETS id when the slash menu opened a named chart. Only the chart
   *  dialog reads it; the other five ignore it. */
  preset?: string;
};

// All insert dialogs share this exact prop contract (see each file's own
// `{ editor, open, onOpenChange }` signature) — one lookup instead of seven
// near-identical conditionally-rendered blocks.
const EMBED_DIALOGS: Record<EmbedKind, ComponentType<EmbedDialogProps>> = {
  figure: InsertFigureDialog,
  molecule: InsertMoleculeDialog,
  structureGrid: InsertStructureGridDialog,
  reaction: InsertReactionDialog,
  protein: InsertProteinDialog,
  sequence: InsertSequenceDialog,
  chart: InsertChartDialog,
  math: InsertMathDialog,
};

const lowlight = createLowlight(common);

/** Shared by FileHandler's drop and paste paths: upload, then insert a
 *  figureImage at pos (the drop point) or the selection (paste). A failed
 *  upload just toasts — same retry story as the figure dialog. */
async function insertImageFile(editor: Editor, file: File, pos?: number) {
  try {
    const { sha256, mime } = await uploadBlob(file);
    if (editor.isDestroyed) return;
    const node = { type: "figureImage", attrs: { blobKey: sha256, mime } };
    const chain = editor.chain().focus();
    (pos == null ? chain.insertContent(node) : chain.insertContentAt(pos, node)).run();
  } catch {
    toast.error("Couldn't upload the image. Try again.");
  }
}

// Full-bleed document surface — unlike tasks' boxed comment editor
// (EDITOR_PROSE_CLASS), a page fills the window: no border, no min-height cap;
// flex-1 stretches the prose to the scroller so clicking anywhere focuses it.
const PAGE_PROSE_CLASS = "tiptap flex-1 px-4 py-3 focus:outline-none";

export function PageEditor({
  initial,
  onChange,
  editable = true,
  diff,
  pageId,
}: {
  initial: PMDoc | null;
  onChange: (doc: PMDoc) => void;
  editable?: boolean;
  diff?: DiffProp;
  pageId?: string;
}) {
  const [dialog, setDialog] = useState<{
    kind: EmbedKind;
    editing?: EmbedEditRequest;
    preset?: string;
  } | null>(null);
  // Stable for the component's lifetime, so it can be handed to the slash menu
  // (which captures it once, at extension-build time) and to the embed context.
  const openDialog = useCallback<OpenEmbedDialog>(
    (kind, editing, preset) => setDialog({ kind, editing, preset }),
    [],
  );

  // Math nodes have no React node view (the extension renders KaTeX itself), so
  // a click can't hand the dialog a node view's own updateAttributes like other
  // embeds — and no live editor exists yet where these closures are built. The
  // click passes latex/kind/pos instead, and the math dialog runs
  // updateInlineMath/updateBlockMath itself via its `editor` prop; onUpdate is
  // a required-by-type stub it never calls. `editable` is static per mount
  // (view and edit surfaces are separate mounts), so capturing it is safe.
  const openMathEdit = (kind: "inline" | "block", node: PMNode, pos: number) => {
    if (!editable) return; // clicks still land on read-only surfaces (page view, history)
    openDialog("math", {
      attrs: { latex: node.attrs.latex, mathKind: kind, pos },
      onUpdate: () => {},
    });
  };

  const editor = useEditor({
    immediatelyRender: false, // Next SSR: render on the client to avoid hydration drift
    editable,
    // Entering edit mode (incl. landing here right after creating a page) should
    // put the caret in the doc — without this the user has to click first.
    autofocus: editable && "end",
    extensions: [
      ...sharedExtensions({
        placeholder: "Write up the experiment… type @ people, # entities, / to insert",
        // Own Document (attrs) and a lowlight code block replace StarterKit's.
        starterKit: { document: false, codeBlock: false },
        mentions: [
          { char: "@", pluginName: "mention", items: searchMemberItems },
          {
            char: "#",
            pluginName: "entityMention",
            items: searchEntities,
            // Entities get a chip (EntityLink), not the default mention node —
            // label freezes at insert time; see nodes/entity-link.ts.
            onSelect: (editor, range, item) =>
              editor
                .chain()
                .focus()
                .insertContentAt(range, [
                  {
                    type: "entityLink",
                    attrs: { entityType: item.kind, entityId: item.id, label: item.label },
                  },
                  { type: "text", text: " " },
                ])
                .run(),
          },
        ],
      }),
      PageDoc,
      Table.configure({ resizable: true }),
      TableRow,
      TableHeader,
      TableCell,
      Image,
      CodeBlockLowlight.configure({ lowlight }),
      Superscript,
      Subscript,
      Highlight,
      // Collapsible sections. Default persist:false keeps open/closed state out
      // of the doc — toggling folds is browsing, not editing, so it must not
      // dirty the page or show up in version history.
      Details,
      DetailsSummary,
      DetailsContent,
      FileHandler.configure({
        allowedMimeTypes: ["image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"],
        onDrop: (editor, files, pos) => {
          for (const f of files) void insertImageFile(editor, f, pos);
        },
        // htmlContent means the paste is rich content that merely references
        // images (e.g. copied from a web page) — the normal HTML paste path
        // handles that; inserting the files too would duplicate them.
        onPaste: (editor, files, htmlContent) => {
          if (htmlContent) return;
          for (const f of files) void insertImageFile(editor, f);
        },
      }),
      // Also gives input rules: $$x^2$$ → inline math, $$$…$$$ → block math.
      Mathematics.configure({
        katexOptions: { throwOnError: false },
        inlineOptions: { onClick: (node, pos) => openMathEdit("inline", node, pos) },
        blockOptions: { onClick: (node, pos) => openMathEdit("block", node, pos) },
      }),
      FigureImage,
      ChemStructure,
      StructureGrid,
      Chart,
      ReactionScheme,
      ProteinStructure,
      SequenceViewer,
      EntityLink,
      // openDialog is useCallback([])-stable, so — unlike a value captured from
      // props/other state — no ref indirection is needed to keep this pointed at
      // the current one across re-renders.
      createSlashMenu(openDialog),
      ...(diff ? [diffHighlight(diff.ranges, diff.side)] : []),
    ],
    content: (initial as Content) ?? "",
    editorProps: { attributes: { class: PAGE_PROSE_CLASS } },
    onUpdate: ({ editor }) => onChange(editor.getJSON() as PMDoc),
  });

  // Embed dialogs unmount on close (EMBED_DIALOGS below renders only the open
  // one), which cuts off Radix's own close-time focus restore — after
  // Cancel/Esc the caret would land on <body> and the user has to click back
  // into the editor. Put focus back ourselves once the dialog is gone.
  const hadDialog = useRef(false);
  useEffect(() => {
    if (hadDialog.current && !dialog && editor && !editor.isDestroyed) editor.commands.focus();
    hadDialog.current = !!dialog;
  }, [dialog, editor]);

  // Re-renders the width/text-size wrapper when the settings menu edits the
  // doc attrs; for read-only surfaces the attrs are static from `initial`.
  const settings = useEditorState({
    editor,
    selector: (ctx) =>
      pageSettings(
        ctx.editor ? { type: "doc", attrs: ctx.editor.state.doc.attrs } : (initial ?? null),
      ),
  });

  if (!editor) return null;

  // Lazily mounted: only the dialog matching `dialog.kind` ever renders, and it
  // unmounts (not just closes) once dismissed — which is also what lets each
  // dialog seed its form state straight from `editing.attrs` at mount, with no
  // prop-sync effect.
  const EmbedDialog = dialog ? EMBED_DIALOGS[dialog.kind] : null;

  return (
    <>
      {/* Wraps EditorContent because TipTap's node-view portals are created
          inside it — that's how EmbedToolbar reaches openDialog. */}
      <EmbedDialogProvider value={openDialog}>
        {editable && <EditorToolbar editor={editor} onInsert={openDialog} pageId={pageId} />}
        {/* flex column so PAGE_PROSE_CLASS's flex-1 can stretch the prose to fill
            the scroller (percentage min-heights don't resolve here; flex does) */}
        <TableContextMenu editor={editor}>
          <EditorContent
            editor={editor}
            className={cn(
              "flex min-h-0 flex-1 flex-col",
              settings && pageSettingsClasses(settings),
            )}
          />
        </TableContextMenu>
      </EmbedDialogProvider>
      {EmbedDialog && (
        <EmbedDialog
          editor={editor}
          open={true}
          editing={dialog?.editing}
          preset={dialog?.preset}
          onOpenChange={(open) => !open && setDialog(null)}
        />
      )}
    </>
  );
}
