"use client";

import { useQuery } from "@tanstack/react-query";
import type { Editor } from "@tiptap/react";
import { useState } from "react";

import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { cn } from "@/shared/lib/utils";

import type { EmbedEditRequest } from "../embed-toolbar";
import { MoleculeDepiction, computeChemSnapshot } from "../renderers/molecule-depiction";
import { EmbedDialogFrame } from "./embed-dialog-frame";

// Fixed per pane so switching tabs never resizes the dialog (no layout jump).
const PANE_CLASS = "flex h-64 flex-col gap-4";

// Every preview state — empty prompt, lookup in flight, error, resolved
// depiction — occupies this same box, so neither typing nor switching tabs
// resizes the dialog.
const PREVIEW_CLASS =
  "flex h-44 shrink-0 items-center justify-center rounded-md border border-border bg-muted/30";

const CHEMBL_ID_RE = /^CHEMBL\d+$/i;

/**
 * Resolve a ChEMBL ID to its structure. ChEMBL's data API is public and sends
 * `access-control-allow-origin: *`, so this runs client-side with no backend —
 * the same shape as the protein dialog's RCSB lookup. Throws on a miss so
 * TanStack Query can surface it as an error state.
 */
async function fetchChemblMolecule(id: string): Promise<{ smiles: string; name: string | null }> {
  const res = await fetch(`https://www.ebi.ac.uk/chembl/api/data/molecule/${id}.json`);
  if (!res.ok) throw new Error("not found");
  const data = (await res.json()) as {
    pref_name?: string | null;
    molecule_structures?: { canonical_smiles?: string } | null;
  };
  const smiles = data.molecule_structures?.canonical_smiles;
  if (!smiles) throw new Error("no structure");
  return { smiles, name: data.pref_name ?? null };
}

/** Insert-a-chemStructure-node dialog: a pasted SMILES, or a ChEMBL ID looked up
 *  straight from EBI. A node stored with `source: "chemcellar"` (pasted from a
 *  sister app) still renders and edits; it reopens on the SMILES tab. */
export function InsertMoleculeDialog({
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
  // Seeded at mount from the node being edited (see EmbedDialogFrame: every open
  // is a fresh mount, so no prop-sync effect is needed).
  const editingSource = editing?.attrs.source as string | undefined;
  const [tab, setTab] = useState(editingSource === "chembl" ? editingSource : "smiles");
  const [smiles, setSmiles] = useState((editing?.attrs.smiles as string) ?? "");
  const [chemblId, setChemblId] = useState(
    editingSource === "chembl" ? ((editing?.attrs.sourceId as string) ?? "") : "",
  );
  const [caption, setCaption] = useState((editing?.attrs.caption as string) ?? "");
  const [inserting, setInserting] = useState(false);

  const trimmedId = chemblId.trim().toUpperCase();
  const validId = CHEMBL_ID_RE.test(trimmedId);

  // useQuery rather than a hand-rolled effect: it gives the loading/error states
  // and per-ID caching for free, and matches how the rest of the app fetches.
  const chembl = useQuery({
    queryKey: ["chembl", trimmedId],
    queryFn: () => fetchChemblMolecule(trimmedId),
    enabled: tab === "chembl" && validId,
    retry: false,
    staleTime: Number.POSITIVE_INFINITY, // a ChEMBL entry's structure doesn't change under us
  });

  // Whichever tab is active decides what actually gets inserted.
  const resolvedSmiles = tab === "chembl" ? (chembl.data?.smiles ?? "") : smiles.trim();

  async function confirm() {
    if (!resolvedSmiles) return;
    setInserting(true);
    const snapshot = await computeChemSnapshot(resolvedSmiles);
    if (editor.isDestroyed) return;

    // ChEMBL's preferred name makes the provenance line read as more than a bare
    // formula. A pasted SMILES has no name to offer.
    const name = tab === "chembl" ? chembl.data?.name : null;
    const sourceId = tab === "chembl" ? trimmedId : null;

    const attrs = {
      source: tab,
      smiles: resolvedSmiles,
      sourceId,
      snapshot: name ? { ...snapshot, name } : snapshot,
      caption: caption.trim() || null,
      capturedAt: new Date().toISOString(),
    };
    if (editing) {
      editing.onUpdate(attrs);
    } else {
      editor.chain().focus().insertContent({ type: "chemStructure", attrs }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit molecule" : "Insert molecule"}
      description={
        editing
          ? "Change the structure or its caption."
          : "Add a 2D chemical structure from SMILES or a ChEMBL ID."
      }
      confirmLabel={editing ? "Save changes" : "Insert molecule"}
      busyLabel={editing ? "Saving…" : "Inserting…"}
      busy={inserting}
      canConfirm={resolvedSmiles.length > 0}
      onConfirm={() => void confirm()}
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="smiles">SMILES</TabsTrigger>
          <TabsTrigger value="chembl">ChEMBL ID</TabsTrigger>
        </TabsList>

        <TabsContent value="smiles" className={PANE_CLASS}>
          <Field>
            <FieldLabel htmlFor="insert-molecule-smiles">SMILES</FieldLabel>
            <Input
              id="insert-molecule-smiles"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="CCO"
            />
          </Field>
          {/* Fixed height, not flex-1: the preview's content differs per state
              (depiction / message) and per tab, and a content-sized box would
              resize the dialog as you type. */}
          <div className={cn(PREVIEW_CLASS, "text-sm text-muted-foreground")}>
            {/* Guarded rather than letting MoleculeDepiction handle it: with no
                SMILES it has nothing to render and sits on its loading skeleton
                forever, so an empty field reads as "still working" instead of
                "waiting for you". Same prompt-then-preview shape as ChEMBL. */}
            {smiles.trim() ? (
              <MoleculeDepiction smiles={smiles.trim()} width={220} height={160} />
            ) : (
              <p>Enter a SMILES string to preview</p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="chembl" className={PANE_CLASS}>
          <Field>
            <FieldLabel htmlFor="insert-molecule-chembl">ChEMBL ID</FieldLabel>
            <Input
              id="insert-molecule-chembl"
              value={chemblId}
              onChange={(e) => setChemblId(e.target.value)}
              placeholder="CHEMBL25"
            />
          </Field>
          <div className={cn(PREVIEW_CLASS, "flex-col gap-1 text-sm text-muted-foreground")}>
            {!validId ? (
              <p>Enter a ChEMBL ID to look it up</p>
            ) : chembl.isPending ? (
              <p>Looking up {trimmedId}…</p>
            ) : chembl.isError ? (
              <p>Couldn&apos;t find {trimmedId} in ChEMBL.</p>
            ) : (
              <>
                <MoleculeDepiction smiles={chembl.data.smiles} width={220} height={130} />
                {chembl.data.name ? <p>{chembl.data.name}</p> : null}
              </>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Outside the tabs: a caption belongs to the inserted figure, not to the
          source it came from, so it shouldn't be duplicated per pane. */}
      <Field>
        <FieldLabel htmlFor="insert-molecule-caption">Caption (optional)</FieldLabel>
        <Input
          id="insert-molecule-caption"
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          placeholder="What this structure shows"
        />
      </Field>
    </EmbedDialogFrame>
  );
}
