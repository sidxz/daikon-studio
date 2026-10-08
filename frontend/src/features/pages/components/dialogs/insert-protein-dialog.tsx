"use client";

import type { Editor } from "@tiptap/react";
import { useState } from "react";

import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";

import { EMBED_WIDTH_CLASS, type EmbedEditRequest, type EmbedWidth } from "../embed-toolbar";
import { StructureViewer } from "../renderers/structure-viewer";
import { EmbedDialogFrame } from "./embed-dialog-frame";

// Classic 4-character PDB code: a digit 1-9 followed by 3 alphanumerics.
const PDB_ID_RE = /^[1-9][A-Z0-9]{3}$/;

const WIDTH_OPTIONS: { value: EmbedWidth; label: string }[] = [
  { value: "full", label: "Full width" },
  { value: "half", label: "Half width" },
  { value: "fit", label: "Small" },
];

// Fixed per pane so switching tabs never resizes the dialog (no layout jump).
// Taller than the molecule/reaction panes to fit the 360px live viewer.
// Tall enough for both fields plus the 360px viewer slot without scrolling;
// the max-h guard only kicks in (and scrolls) on short viewports.
const PANE_CLASS = "flex h-[33rem] max-h-[65vh] flex-col gap-4 overflow-y-auto";

/**
 * Best-effort snapshot metadata (method/resolution) from RCSB — never
 * throws; a failure just leaves a name-only caption, same contract as
 * computeChemSnapshot. ponytail: snapshot metadata best-effort.
 */
async function fetchRcsbSnapshot(id: string): Promise<{ method?: string; resolution?: string }> {
  try {
    const res = await fetch(`https://data.rcsb.org/rest/v1/core/entry/${id}`);
    if (!res.ok) return {};
    const data = (await res.json()) as {
      exptl?: Array<{ method?: string }>;
      rcsb_entry_info?: { resolution_combined?: number[] };
    };
    const resolution = data.rcsb_entry_info?.resolution_combined?.[0];
    return {
      method: data.exptl?.[0]?.method,
      resolution: resolution != null ? `${resolution.toFixed(2)} Å` : undefined,
    };
  } catch {
    return {};
  }
}

/** Insert-a-proteinStructure-node dialog. PDB ID is the only live source in v1
 *  — ProtCellar search is a shown-but-disabled tab so the roadmap is visible
 *  without a dead-end. */
export function InsertProteinDialog({
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
  const [pdbId, setPdbId] = useState((editing?.attrs.sourceId as string) ?? "");
  const [width, setWidth] = useState<EmbedWidth>((editing?.attrs.width as EmbedWidth) ?? "full");
  const [inserting, setInserting] = useState(false);

  const trimmed = pdbId.trim().toUpperCase();
  const valid = PDB_ID_RE.test(trimmed);

  async function confirm() {
    if (!valid) return;
    setInserting(true);
    // Skip the round-trip when only the size changed — the snapshot still describes
    // the same entry.
    const unchangedId = editing?.attrs.sourceId === trimmed;
    const meta = unchangedId ? null : await fetchRcsbSnapshot(trimmed);
    if (editor.isDestroyed) return;
    if (editing) {
      editing.onUpdate({
        sourceId: trimmed,
        width,
        ...(unchangedId
          ? {}
          : { snapshot: { name: trimmed, ...meta }, capturedAt: new Date().toISOString() }),
      });
    } else {
      editor
        .chain()
        .focus()
        .insertContent({
          type: "proteinStructure",
          attrs: {
            source: "pdb",
            sourceId: trimmed,
            chains: null,
            style: null,
            width,
            snapshot: { name: trimmed, ...meta },
            capturedAt: new Date().toISOString(),
          },
        })
        .run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit protein structure" : "Insert protein structure"}
      description={
        editing
          ? "Change the PDB entry or the size of this structure."
          : "Add a 3D structure from a PDB entry."
      }
      confirmLabel={editing ? "Save changes" : "Insert structure"}
      busyLabel={editing ? "Saving…" : "Inserting…"}
      busy={inserting}
      canConfirm={valid}
      onConfirm={() => void confirm()}
    >
      <Tabs defaultValue="pdb">
        <TabsList>
          <TabsTrigger value="pdb">PDB ID</TabsTrigger>
          <TabsTrigger value="protcellar" disabled>
            ProtCellar search
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pdb" className={PANE_CLASS}>
          <Field>
            <FieldLabel htmlFor="insert-protein-pdb-id">PDB ID</FieldLabel>
            <Input
              id="insert-protein-pdb-id"
              value={pdbId}
              onChange={(e) => setPdbId(e.target.value)}
              placeholder="1CRN"
              maxLength={4}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="insert-protein-width">Size</FieldLabel>
            <Select value={width} onValueChange={(v) => setWidth(v as EmbedWidth)}>
              <SelectTrigger id="insert-protein-width">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WIDTH_OPTIONS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          {/* Fixed-height slot: the size options have different heights, and
              without this the dialog would resize as you change the picker. */}
          <div className="flex h-[360px] shrink-0 items-start">
            {valid ? (
              <div className={EMBED_WIDTH_CLASS[width]}>
                <StructureViewer source="pdb" sourceId={trimmed} width={width} />
              </div>
            ) : (
              <div className="flex size-full items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground">
                Enter a 4-character PDB code to preview
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent
          value="protcellar"
          className={`${PANE_CLASS} items-center justify-center text-center text-sm text-muted-foreground`}
        >
          <p>Coming soon — cellar link.</p>
          <p>Search ProtCellar&apos;s registered structures once it&apos;s wired in.</p>
        </TabsContent>
      </Tabs>
    </EmbedDialogFrame>
  );
}
