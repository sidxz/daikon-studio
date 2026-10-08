import { PluginKey } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/react";
import { Extension } from "@tiptap/react";
import { Suggestion, type SuggestionOptions } from "@tiptap/suggestion";

import { suggestionPopup } from "@/features/pages/lib/editor/suggestion-popup";

import { CHART_PRESETS } from "./chart-presets";
import type { OpenEmbedDialog } from "./embed-toolbar";
import { SlashMenuList } from "./slash-menu-list";

/** The embed dialogs the slash menu can open. page-editor.tsx owns the
 *  matching `openDialog` state and renders the dialogs; this module never
 *  touches React state directly. */
export type EmbedKind =
  | "figure"
  | "molecule"
  | "structureGrid"
  | "reaction"
  | "protein"
  | "sequence"
  | "chart"
  | "math";

export type SlashItem = {
  title: string;
  hint: string;
  keywords?: string[];
  run: (editor: Editor) => void;
};

/**
 * Every `/` command: three generic blocks — table, code block, task list —
 * are already registered node types (StarterKit + Task 7's TaskList/TaskItem),
 * so they insert directly. There is no callout node in the schema; don't
 * offer one. The five embeds hand off to `openDialog` (page-editor.tsx's
 * `setOpenDialog`, passed directly — a useState setter is stable for the
 * component's lifetime, so no ref is needed) instead of inserting — the
 * matching dialog inserts on confirm once the user fills it in.
 */
export function buildSlashItems(openDialog: OpenEmbedDialog): SlashItem[] {
  return [
    {
      title: "Table",
      hint: "3×3 table with a header row",
      keywords: ["table", "grid"],
      run: (editor) =>
        editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run(),
    },
    {
      title: "Code block",
      hint: "Fenced, syntax-highlighted code",
      keywords: ["code", "snippet", "pre"],
      run: (editor) => editor.chain().focus().toggleCodeBlock().run(),
    },
    {
      title: "Task list",
      hint: "Checklist with nested items",
      keywords: ["task", "todo", "checklist"],
      run: (editor) => editor.chain().focus().toggleTaskList().run(),
    },
    {
      title: "Figure",
      hint: "Upload an image with a caption",
      keywords: ["figure", "image", "picture"],
      run: () => openDialog("figure"),
    },
    {
      title: "Molecule",
      hint: "Chemical structure from SMILES",
      keywords: ["molecule", "chemical", "structure", "smiles"],
      run: () => openDialog("molecule"),
    },
    {
      title: "Structure grid",
      hint: "A grid of 2D structures from SMILES, ChEMBL or ChemCellar",
      keywords: ["structures", "grid", "compounds", "molecules", "chemcellar", "chembl", "sar"],
      run: () => openDialog("structureGrid"),
    },
    {
      title: "Reaction",
      hint: "Reaction scheme from reaction SMILES",
      keywords: ["reaction", "scheme", "chemistry"],
      run: () => openDialog("reaction"),
    },
    {
      title: "Protein",
      hint: "3D structure from a PDB entry",
      keywords: ["protein", "structure", "pdb"],
      run: () => openDialog("protein"),
    },
    {
      title: "Sequence",
      hint: "DNA/RNA/protein sequence track",
      keywords: ["sequence", "dna", "rna", "fasta"],
      run: () => openDialog("sequence"),
    },
    {
      title: "Chart",
      hint: "Bar, line, scatter, radar or pie from a pasted table",
      keywords: ["chart", "graph", "plot", "data"],
      run: () => openDialog("chart"),
    },
    {
      title: "Equation",
      hint: "LaTeX math rendered with KaTeX",
      keywords: ["equation", "math", "latex", "formula", "katex"],
      run: () => openDialog("math"),
    },
    {
      title: "Toggle section",
      hint: "Collapsible section that folds its content away",
      keywords: ["toggle", "details", "collapse", "fold", "section", "accordion"],
      run: (editor) => {
        editor.chain().focus().setDetails().run();
        // A new section mounts closed, and with persist:false open-state is
        // DOM-only (no command can open it) — so simulate the toggle click the
        // user would make next. rAF because the node view doesn't exist until
        // the transaction has rendered. Keeps "sections load collapsed" intact:
        // nothing is written to the doc.
        requestAnimationFrame(() => {
          const { node } = editor.view.domAtPos(editor.state.selection.from);
          const el = node instanceof Element ? node : node.parentElement;
          el?.closest('[data-type="details"]')
            ?.querySelector<HTMLButtonElement>(":scope > button")
            ?.click();
        });
      },
    },
    // Each preset is a configuration of the same five marks, so it costs a table
    // row rather than a code path — and a user after a volcano plot searches for
    // "volcano", not "chart".
    ...Object.entries(CHART_PRESETS).map(([id, p]) => ({
      title: p.label,
      hint: p.hint,
      keywords: p.keywords,
      run: () => openDialog("chart", undefined, id),
    })),
  ];
}

/** Case-insensitive match on title or keywords; an empty/blank query returns everything. */
export function filterSlashItems(items: SlashItem[], query: string): SlashItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return items;
  return items.filter(
    (it) => it.title.toLowerCase().includes(q) || it.keywords?.some((k) => k.includes(q)),
  );
}

const SLASH_PLUGIN_KEY = new PluginKey("slashCommand");

/**
 * `/` suggestion extension (`@tiptap/suggestion`, already a dep — same
 * primitive `lib/editor/mention.ts` builds on for `@`/`#`). Selecting an item
 * deletes the typed "/query" range, then runs the item: generic blocks insert
 * immediately; embeds call `openDialog` instead.
 */
export function createSlashMenu(openDialog: OpenEmbedDialog) {
  return Extension.create({
    name: "slashCommand",
    addOptions() {
      return {
        suggestion: {
          char: "/",
          pluginKey: SLASH_PLUGIN_KEY,
          items: ({ query }) => filterSlashItems(buildSlashItems(openDialog), query),
          command: ({ editor, range, props }) => {
            editor.chain().focus().deleteRange(range).run();
            props.run(editor);
          },
          render: suggestionPopup<SlashItem, SlashItem>(SlashMenuList),
        } satisfies Partial<SuggestionOptions<SlashItem, SlashItem>>,
      };
    },
    addProseMirrorPlugins() {
      return [Suggestion({ editor: this.editor, ...this.options.suggestion })];
    },
  });
}
