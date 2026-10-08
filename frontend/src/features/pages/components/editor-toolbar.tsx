"use client";

import { type Editor, useEditorState } from "@tiptap/react";
import {
  Bold,
  ChevronDown,
  Code,
  Highlighter,
  Italic,
  Link as LinkIcon,
  List,
  ListOrdered,
  ListTodo,
  Minus,
  Plus,
  Redo2,
  SquareCode,
  Strikethrough,
  Subscript as SubscriptIcon,
  Superscript as SuperscriptIcon,
  Table as TableIcon,
  TextQuote,
  Underline,
  Undo2,
} from "lucide-react";
import { type ReactNode, useState } from "react";

import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import { Input } from "@/shared/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import { cn } from "@/shared/lib/utils";

import { PageSettingsMenu } from "./page-settings-menu";
import type { EmbedKind } from "./slash-menu";

/**
 * The Insert menu, in display order (string keys keep insertion order).
 *
 * A Record rather than a list of pairs, on purpose: as a list, adding a new
 * EmbedKind and forgetting to offer it here compiled fine and the embed was
 * reachable only by typing `/`. As a Record the object literal fails to compile
 * until every kind has a label, so the menu cannot silently fall behind again.
 */
const EMBED_LABELS: Record<EmbedKind, string> = {
  figure: "Figure",
  molecule: "Molecule",
  structureGrid: "Structure grid",
  reaction: "Reaction scheme",
  protein: "Protein structure",
  sequence: "Sequence",
  chart: "Chart",
  math: "Equation",
};

// Object.entries widens the key to string; the Record above is what guarantees
// these really are EmbedKinds.
const EMBED_MENU = Object.entries(EMBED_LABELS) as [EmbedKind, string][];

const HEADING_LEVELS = [1, 2, 3] as const;

function ToolButton({
  label,
  active = false,
  disabled = false,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      className={cn(active && "bg-muted text-foreground")}
    >
      {children}
    </Button>
  );
}

function Divider() {
  return <span aria-hidden className="mx-1 h-5 w-px shrink-0 bg-border" />;
}

const GRID_COLS = 8;
const GRID_ROWS = 8;

/** Word/TipTap-style size picker: hover (or arrow-key focus) the grid to choose
 *  the shape, with a live "cols x rows" readout, then click to insert. Every cell
 *  is a real button carrying an aria-label, so this is reachable by keyboard
 *  rather than being a mouse-only affordance. */
function TableGridPicker({ editor }: { editor: Editor }) {
  const [open, setOpen] = useState(false);
  const [size, setSize] = useState({ cols: 0, rows: 0 });

  const insert = (cols: number, rows: number) => {
    // withHeaderRow means the first row is <th>; the body rows are what's left.
    editor.chain().focus().insertTable({ rows, cols, withHeaderRow: true }).run();
    setOpen(false);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (o) setSize({ cols: 0, rows: 0 });
      }}
    >
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Insert table"
          title="Insert table"
        >
          <TableIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-auto p-2">
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `repeat(${GRID_COLS}, 1.25rem)` }}
          onMouseLeave={() => setSize({ cols: 0, rows: 0 })}
        >
          {Array.from({ length: GRID_ROWS * GRID_COLS }, (_, i) => {
            const col = (i % GRID_COLS) + 1;
            const row = Math.floor(i / GRID_COLS) + 1;
            const on = col <= size.cols && row <= size.rows;
            return (
              <button
                key={i}
                type="button"
                aria-label={`${col} by ${row} table`}
                className={cn(
                  "size-5 rounded-[3px] border transition-colors",
                  on ? "border-primary bg-primary/25" : "border-border",
                )}
                onMouseEnter={() => setSize({ cols: col, rows: row })}
                onFocus={() => setSize({ cols: col, rows: row })}
                onClick={() => insert(col, row)}
              />
            );
          })}
        </div>
        <p aria-live="polite" className="mt-2 text-center text-sm text-muted-foreground">
          {size.cols > 0 ? `${size.cols} × ${size.rows}` : "Pick a size"}
        </p>
      </PopoverContent>
    </Popover>
  );
}

/**
 * A bare "example.com" is a *relative* href to the browser, so such a link would
 * navigate inside the app (/pages/<id>/example.com) instead of out to the site.
 * Anything without a scheme gets https://; "mailto:"/"ftp:" and friends already
 * carry one and pass through untouched (as do unsafe ones like "javascript:",
 * which TipTap's own setLink then refuses).
 */
export function normalizeLinkHref(raw: string): string {
  const url = raw.trim();
  return /^[a-z][a-z0-9+.-]*:/i.test(url) ? url : `https://${url}`;
}

/** URL entry in a popover instead of window.prompt — prompt() blocks the whole
 *  tab, and the explicit Apply/Remove/Cancel buttons are a UI standard here. */
function LinkControl({ editor, active }: { editor: Editor; active: boolean }) {
  const [open, setOpen] = useState(false);
  const [href, setHref] = useState("");
  const [rejected, setRejected] = useState(false);

  const apply = () => {
    if (!href.trim()) return;
    // setLink refuses unsafe schemes (javascript:, data:) by returning false.
    // Staying open with a message beats closing on a link that was never applied.
    const ok = editor
      .chain()
      .focus()
      .extendMarkRange("link")
      .setLink({ href: normalizeLinkHref(href) })
      .run();
    setRejected(!ok);
    if (ok) setOpen(false);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        setRejected(false);
        if (o) setHref((editor.getAttributes("link").href as string) ?? "");
      }}
    >
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Link"
          title="Link"
          aria-pressed={active}
          className={cn(active && "bg-muted text-foreground")}
        >
          <LinkIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 space-y-2 p-3">
        <Input
          autoFocus
          placeholder="https://…"
          value={href}
          onChange={(e) => {
            setHref(e.target.value);
            setRejected(false);
          }}
          onKeyDown={(e) => e.key === "Enter" && apply()}
        />
        {rejected && (
          <p className="text-sm text-destructive">
            That doesn’t look like a web address we can link to.
          </p>
        )}
        <div className="flex justify-end gap-2">
          {active && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                editor.chain().focus().extendMarkRange("link").unsetLink().run();
                setOpen(false);
              }}
            >
              Remove
            </Button>
          )}
          <Button type="button" variant="outline" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button type="button" size="sm" disabled={!href.trim()} onClick={apply}>
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

/** Formatting toolbar for the page editor. Buttons cover what the registered
 *  extensions support (StarterKit incl. link/underline, task list, table, code
 *  block); science embeds reuse the slash-menu dialogs via `onInsert`. */
export function EditorToolbar({
  editor,
  onInsert,
  pageId,
}: {
  editor: Editor;
  onInsert: (kind: EmbedKind) => void;
  pageId?: string;
}) {
  const s = useEditorState({
    editor,
    selector: ({ editor: e }) => ({
      bold: e.isActive("bold"),
      italic: e.isActive("italic"),
      underline: e.isActive("underline"),
      strike: e.isActive("strike"),
      superscript: e.isActive("superscript"),
      subscript: e.isActive("subscript"),
      highlight: e.isActive("highlight"),
      code: e.isActive("code"),
      link: e.isActive("link"),
      bulletList: e.isActive("bulletList"),
      orderedList: e.isActive("orderedList"),
      taskList: e.isActive("taskList"),
      blockquote: e.isActive("blockquote"),
      codeBlock: e.isActive("codeBlock"),
      heading: HEADING_LEVELS.find((level) => e.isActive("heading", { level })) ?? null,
      canUndo: e.can().undo(),
      canRedo: e.can().redo(),
    }),
  });

  const chain = () => editor.chain().focus();

  return (
    <div
      role="toolbar"
      aria-label="Formatting"
      className="sticky top-0 z-10 flex flex-wrap items-center gap-0.5 border-b border-border bg-background px-2 py-1"
    >
      <ToolButton label="Undo" disabled={!s.canUndo} onClick={() => chain().undo().run()}>
        <Undo2 />
      </ToolButton>
      <ToolButton label="Redo" disabled={!s.canRedo} onClick={() => chain().redo().run()}>
        <Redo2 />
      </ToolButton>

      <Divider />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button type="button" variant="ghost" size="sm" className="w-28 justify-between">
            {s.heading ? `Heading ${s.heading}` : "Paragraph"}
            <ChevronDown />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem onSelect={() => chain().setParagraph().run()}>
            Paragraph
          </DropdownMenuItem>
          {HEADING_LEVELS.map((level) => (
            <DropdownMenuItem key={level} onSelect={() => chain().toggleHeading({ level }).run()}>
              Heading {level}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <Divider />

      <ToolButton label="Bold" active={s.bold} onClick={() => chain().toggleBold().run()}>
        <Bold />
      </ToolButton>
      <ToolButton label="Italic" active={s.italic} onClick={() => chain().toggleItalic().run()}>
        <Italic />
      </ToolButton>
      <ToolButton
        label="Underline"
        active={s.underline}
        onClick={() => chain().toggleUnderline().run()}
      >
        <Underline />
      </ToolButton>
      <ToolButton
        label="Strikethrough"
        active={s.strike}
        onClick={() => chain().toggleStrike().run()}
      >
        <Strikethrough />
      </ToolButton>
      <ToolButton
        label="Superscript"
        active={s.superscript}
        onClick={() => chain().toggleSuperscript().run()}
      >
        <SuperscriptIcon />
      </ToolButton>
      <ToolButton
        label="Subscript"
        active={s.subscript}
        onClick={() => chain().toggleSubscript().run()}
      >
        <SubscriptIcon />
      </ToolButton>
      <ToolButton
        label="Highlight"
        active={s.highlight}
        onClick={() => chain().toggleHighlight().run()}
      >
        <Highlighter />
      </ToolButton>
      <ToolButton label="Inline code" active={s.code} onClick={() => chain().toggleCode().run()}>
        <Code />
      </ToolButton>
      <LinkControl editor={editor} active={s.link} />

      <Divider />

      <ToolButton
        label="Bullet list"
        active={s.bulletList}
        onClick={() => chain().toggleBulletList().run()}
      >
        <List />
      </ToolButton>
      <ToolButton
        label="Numbered list"
        active={s.orderedList}
        onClick={() => chain().toggleOrderedList().run()}
      >
        <ListOrdered />
      </ToolButton>
      <ToolButton
        label="Task list"
        active={s.taskList}
        onClick={() => chain().toggleTaskList().run()}
      >
        <ListTodo />
      </ToolButton>

      <Divider />

      <ToolButton
        label="Quote"
        active={s.blockquote}
        onClick={() => chain().toggleBlockquote().run()}
      >
        <TextQuote />
      </ToolButton>
      <ToolButton
        label="Code block"
        active={s.codeBlock}
        onClick={() => chain().toggleCodeBlock().run()}
      >
        <SquareCode />
      </ToolButton>
      {/* Creation only. Row/column editing lives on the table's right-click menu
          (TableContextMenu), since those commands act on the cell you point at. */}
      <TableGridPicker editor={editor} />
      <ToolButton label="Divider" onClick={() => chain().setHorizontalRule().run()}>
        <Minus />
      </ToolButton>

      <Divider />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button type="button" variant="ghost" size="sm">
            <Plus />
            Insert
            <ChevronDown />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {EMBED_MENU.map(([kind, label]) => (
            <DropdownMenuItem key={kind} onSelect={() => onInsert(kind)}>
              {label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <PageSettingsMenu editor={editor} />
    </div>
  );
}
