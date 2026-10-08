"use client";

import type { Editor } from "@tiptap/react";
import { useState } from "react";

import { Field, FieldLabel } from "@/shared/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Textarea } from "@/shared/components/ui/textarea";

import type { EmbedEditRequest } from "../embed-toolbar";
import { SequenceTrack } from "../renderers/sequence-track";
import { EmbedDialogFrame } from "./embed-dialog-frame";

// Fixed height so typing/pasting never resizes the dialog (no layout jump) —
// same recipe as the other insert-*-dialogs.
const PANE_CLASS = "flex h-[28rem] flex-col gap-4 overflow-y-auto";

// Letters only once whitespace/newlines are stripped — a pasted FASTA-ish
// blob (blank lines, wrapped rows) is the expected shape, not one bare line.
// `+` also rejects the empty string, so this doubles as the non-empty check.
const SEQUENCE_RE = /^[A-Za-z]+$/;

/** Insert-a-sequenceViewer-node dialog. No async step (no fetch/WASM), so
 *  there's no busy state to manage. Feature annotation is a later pass: v1
 *  always inserts `features: []`. */
export function InsertSequenceDialog({
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
  const [seqType, setSeqType] = useState<"protein" | "dna">(
    (editing?.attrs.seqType as "protein" | "dna") ?? "protein",
  );
  const [raw, setRaw] = useState((editing?.attrs.sequence as string) ?? "");

  const cleaned = raw.replace(/\s+/g, "");
  const valid = SEQUENCE_RE.test(cleaned);

  function confirm() {
    if (!valid || editor.isDestroyed) return;
    const attrs = { seqType, sequence: cleaned, features: [], source: null, sourceId: null };
    if (editing) {
      editing.onUpdate(attrs);
    } else {
      editor.chain().focus().insertContent({ type: "sequenceViewer", attrs }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit sequence" : "Insert sequence"}
      description={
        editing
          ? "Change this sequence or its type."
          : "Add a protein or DNA sequence as an annotated track."
      }
      confirmLabel={editing ? "Save changes" : "Insert sequence"}
      canConfirm={valid}
      onConfirm={confirm}
      className="sm:max-w-lg"
    >
      <div className={PANE_CLASS}>
        <Field>
          <FieldLabel htmlFor="insert-sequence-type">Sequence type</FieldLabel>
          <Select value={seqType} onValueChange={(v) => setSeqType(v as "protein" | "dna")}>
            <SelectTrigger id="insert-sequence-type" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="protein">Protein</SelectItem>
              <SelectItem value="dna">DNA</SelectItem>
            </SelectContent>
          </Select>
        </Field>

        <Field>
          <FieldLabel htmlFor="insert-sequence-textarea">Sequence</FieldLabel>
          <Textarea
            id="insert-sequence-textarea"
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
            placeholder="MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSER…"
            className="min-h-32 max-h-32 resize-none overflow-y-auto font-mono text-sm"
          />
        </Field>

        <div className="flex h-40 w-full overflow-y-auto rounded-md border border-border bg-muted/30 p-2">
          {valid ? (
            <SequenceTrack seqType={seqType} sequence={cleaned} features={[]} />
          ) : (
            <p className="flex h-full w-full items-center justify-center text-center text-sm text-muted-foreground">
              {raw ? "Letters only — no digits or punctuation." : "Paste a sequence to preview"}
            </p>
          )}
        </div>
      </div>
    </EmbedDialogFrame>
  );
}
