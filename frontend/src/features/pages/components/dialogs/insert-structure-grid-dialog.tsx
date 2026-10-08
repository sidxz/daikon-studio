"use client";

import type { Editor } from "@tiptap/react";
import { useState } from "react";

import {
  MAX_STRUCTURES,
  type StructureItem,
  capStructures,
  fetchChemblStructures,
  parseStructureList,
} from "@/features/pages/lib/molecules";
import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { Textarea } from "@/shared/components/ui/textarea";
import { cn } from "@/shared/lib/utils";

import type { EmbedEditRequest } from "../embed-toolbar";
import { MoleculeDepiction } from "../renderers/molecule-depiction";
import { EmbedDialogFrame } from "./embed-dialog-frame";

/**
 * The tab strip plus one pane, at a fixed height, so switching tabs never
 * resizes the dialog (no layout jump).
 *
 * The height has to live HERE, on the Tabs root, not on the panes. Tabs renders
 * a `flex flex-col` root and gives every TabsContent `flex-1` — that is
 * `flex-basis: 0%`, which overrides the `height` property in a column flex
 * container, so an `h-*` on a pane is silently inert and each pane sizes to its
 * own content instead. Pinning the root turns that same `flex-1` into the thing
 * that makes all panes equal.
 */
const TABS_CLASS = "h-[23rem]";

// min-h-0 so the pane may shrink inside the pinned root, and overflow scrolls
// within the pane rather than pushing the dialog taller.
const PANE_CLASS = "flex min-h-0 flex-col gap-3 overflow-y-auto";
const PREVIEW_CLASS =
  "flex h-32 shrink-0 items-center gap-2 overflow-x-auto rounded-md border border-border bg-muted/30 p-2";

/** How many cards the preview strip shows before collapsing to a count. */
const PREVIEW_LIMIT = 6;

/** Re-serialise stored items so the SMILES tab can seed itself on edit. There is
 *  no verbatim `raw` attribute to fall back on — see nodes/structure-grid.ts. */
function toPasteText(items: StructureItem[]): string {
  return items.map((i) => `${i.name},${i.smiles}`).join("\n");
}

/** Shared by every tab's preview strip. Renders at most a handful of cards —
 *  this is a "did I paste the right thing" check, not the grid itself. */
function PreviewStrip({ items, empty }: { items: StructureItem[]; empty: string }) {
  if (items.length === 0) {
    return <p className="m-auto text-sm text-muted-foreground">{empty}</p>;
  }
  return (
    <>
      {items.slice(0, PREVIEW_LIMIT).map((item, i) => (
        <div key={`${item.smiles}-${i}`} className="shrink-0 text-center">
          {item.smiles ? (
            <MoleculeDepiction smiles={item.smiles} width={80} height={60} />
          ) : (
            <div className="flex h-[60px] w-20 items-center justify-center text-[10px] text-muted-foreground">
              No structure
            </div>
          )}
          <p className="w-20 truncate text-[10px] text-muted-foreground">{item.name}</p>
        </div>
      ))}
      {items.length > PREVIEW_LIMIT ? (
        <p className="shrink-0 self-center text-xs text-muted-foreground">
          {`+${items.length - PREVIEW_LIMIT} more`}
        </p>
      ) : null}
    </>
  );
}

export function InsertStructureGridDialog({
  editor,
  open,
  onOpenChange,
  editing,
}: {
  editor: Editor;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editing?: EmbedEditRequest;
}) {
  // Seeded at mount from the node being edited — every open is a fresh mount
  // (see EmbedDialogFrame), so no prop-sync effect is needed.
  const editingSource = editing?.attrs.source as string | undefined;
  const editingItems = (editing?.attrs.items as StructureItem[] | undefined) ?? [];

  // Reopens on whichever tab produced the node, so editing never starts by
  // making the author find their own source again.
  // ChemCellar-sourced grids (from a sister app) reopen on the pasted list.
  const [tab, setTab] = useState(editingSource === "chembl" ? "chembl" : "smiles");
  const [paste, setPaste] = useState(editingSource === "chembl" ? "" : toPasteText(editingItems));
  const [chemblText, setChemblText] = useState(
    editingSource === "chembl" ? editingItems.map((i) => i.id).join("\n") : "",
  );
  // Seeded from the stored items when editing, so reopening a ChEMBL grid shows
  // its structures and can be saved straight away. Left null on a fresh insert.
  // Without this, changing only the caption would force a pointless round trip
  // to EBI just to re-enable the confirm button.
  const [chembl, setChembl] = useState<{
    items: StructureItem[];
    missing: string[];
    dropped: number;
  } | null>(editingSource === "chembl" ? { items: editingItems, missing: [], dropped: 0 } : null);
  const [chemblError, setChemblError] = useState<string | null>(null);
  const [resolving, setResolving] = useState(false);
  const [caption, setCaption] = useState((editing?.attrs.caption as string) ?? "");

  const pasted = capStructures(parseStructureList(paste));
  const resolved = tab === "chembl" ? (chembl?.items ?? []) : pasted.items;
  const dropped = tab === "chembl" ? (chembl?.dropped ?? 0) : tab === "smiles" ? pasted.dropped : 0;

  const chemblIds = chemblText
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);

  async function lookUpChembl() {
    setResolving(true);
    setChemblError(null);
    try {
      setChembl(await fetchChemblStructures(chemblIds));
    } catch {
      setChembl(null);
      setChemblError("Couldn't reach ChEMBL. Check the ids and try again.");
    } finally {
      setResolving(false);
    }
  }

  function confirm() {
    if (resolved.length === 0) return;
    const attrs = {
      source: tab,
      items: resolved,
      sourceId: null,
      caption: caption.trim() || null,
      capturedAt: new Date().toISOString(),
    };
    if (editing) {
      editing.onUpdate(attrs);
    } else {
      editor.chain().focus().insertContent({ type: "structureGrid", attrs }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit structure grid" : "Insert structure grid"}
      description={
        editing
          ? "Change what the grid contains, or its caption."
          : "A grid of 2D structures from a pasted list or ChEMBL ids."
      }
      confirmLabel={editing ? "Save changes" : "Insert grid"}
      busyLabel="Looking up…"
      busy={resolving}
      canConfirm={resolved.length > 0}
      onConfirm={confirm}
      className="sm:max-w-2xl"
    >
      <Tabs value={tab} onValueChange={setTab} className={TABS_CLASS}>
        <TabsList>
          <TabsTrigger value="smiles">SMILES list</TabsTrigger>
          <TabsTrigger value="chembl">ChEMBL ids</TabsTrigger>
        </TabsList>

        <TabsContent value="smiles" className={PANE_CLASS}>
          <Field>
            <FieldLabel htmlFor="grid-paste">One compound per line: name, SMILES</FieldLabel>
            <Textarea
              id="grid-paste"
              value={paste}
              onChange={(e) => setPaste(e.target.value)}
              rows={6}
              placeholder={
                "Erlotinib,COCCOc1cc2c(Nc3cccc(C#C)c3)ncnc2cc1OCCOC\n" +
                "Gefitinib,COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
              }
            />
          </Field>
          <div className={cn(PREVIEW_CLASS)}>
            <PreviewStrip items={pasted.items} empty="Paste a list to preview it" />
          </div>
        </TabsContent>

        <TabsContent value="chembl" className={PANE_CLASS}>
          <Field>
            <FieldLabel htmlFor="grid-chembl">
              ChEMBL ids, one per line or comma separated
            </FieldLabel>
            <Textarea
              id="grid-chembl"
              value={chemblText}
              onChange={(e) => setChemblText(e.target.value)}
              rows={4}
              placeholder={"CHEMBL25\nCHEMBL192"}
            />
          </Field>
          {/* An explicit button, not lookup-as-you-type: the ids arrive by paste,
              and firing a request per keystroke would hammer EBI mid-paste. */}
          <button
            type="button"
            className="self-start text-sm underline underline-offset-4 disabled:opacity-50"
            disabled={chemblIds.length === 0 || resolving}
            onClick={() => void lookUpChembl()}
          >
            {resolving ? "Looking up…" : `Look up ${chemblIds.length} id(s)`}
          </button>
          <div className={cn(PREVIEW_CLASS)}>
            <PreviewStrip
              items={chembl?.items ?? []}
              empty={chemblError ?? "Enter ids and look them up"}
            />
          </div>
          {chembl?.missing.length ? (
            // Named, not silently dropped: a typo'd id should be fixable.
            <p className="text-xs text-muted-foreground">
              {`Not found in ChEMBL: ${chembl.missing.join(", ")}`}
            </p>
          ) : null}
        </TabsContent>
      </Tabs>

      {dropped > 0 ? (
        // One template literal: JSX strips the whitespace around an expression
        // that falls at a line break, which previously rendered "5are".
        <p className="text-xs text-muted-foreground">
          {`Showing the first ${MAX_STRUCTURES}; ${dropped} more won't be included.`}
        </p>
      ) : null}

      <Field>
        <FieldLabel htmlFor="grid-caption">Caption (optional)</FieldLabel>
        <Input
          id="grid-caption"
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          placeholder="What this set shows"
        />
      </Field>
    </EmbedDialogFrame>
  );
}
