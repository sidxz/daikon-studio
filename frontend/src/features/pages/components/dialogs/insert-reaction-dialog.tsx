"use client";

import type { Editor } from "@tiptap/react";
import { useState } from "react";

import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";

import type { EmbedEditRequest } from "../embed-toolbar";
import { ReactionDepiction } from "../renderers/reaction-depiction";
import { EmbedDialogFrame } from "./embed-dialog-frame";

/** Insert-a-reactionScheme-node dialog. Unlike the molecule/figure/protein
 *  dialogs, confirm() has no async step — `reactionScheme` carries no
 *  RDKit-computed snapshot — so there's no busy state to manage. */
export function InsertReactionDialog({
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
  const [reactionSmiles, setReactionSmiles] = useState(
    (editing?.attrs.reactionSmiles as string) ?? "",
  );
  const [title, setTitle] = useState((editing?.attrs.title as string) ?? "");
  const [conditions, setConditions] = useState((editing?.attrs.conditions as string) ?? "");

  const trimmed = reactionSmiles.trim();

  function confirm() {
    if (!trimmed || editor.isDestroyed) return;
    const attrs = {
      reactionSmiles: trimmed,
      title: title.trim() || null,
      conditions: conditions.trim() || null,
    };
    if (editing) {
      editing.onUpdate(attrs);
    } else {
      editor.chain().focus().insertContent({ type: "reactionScheme", attrs }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit reaction" : "Insert reaction"}
      description={
        editing ? "Change this reaction scheme." : "Add a reaction scheme from reaction SMILES."
      }
      confirmLabel={editing ? "Save changes" : "Insert reaction"}
      canConfirm={!!trimmed}
      onConfirm={confirm}
    >
      {/* Fixed height so filling in fields never resizes the dialog — tall
          enough that everything fits without scrolling; the max-h guard only
          kicks in (and scrolls) on short viewports. */}
      <div className="flex h-[26rem] max-h-[65vh] flex-col gap-4 overflow-y-auto">
        <Field>
          <FieldLabel htmlFor="insert-reaction-smiles">Reaction SMILES</FieldLabel>
          <Input
            id="insert-reaction-smiles"
            value={reactionSmiles}
            onChange={(e) => setReactionSmiles(e.target.value)}
            placeholder="CC(=O)O.CCO>>CC(=O)OCC.O"
          />
        </Field>

        <Field>
          <FieldLabel htmlFor="insert-reaction-title">Title (optional)</FieldLabel>
          <Input
            id="insert-reaction-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Fischer esterification"
          />
        </Field>

        <Field>
          <FieldLabel htmlFor="insert-reaction-conditions">Conditions (optional)</FieldLabel>
          <Input
            id="insert-reaction-conditions"
            value={conditions}
            onChange={(e) => setConditions(e.target.value)}
            placeholder="H2SO4, reflux"
          />
        </Field>

        <div className="flex flex-1 items-center justify-center rounded-md border border-border bg-muted/30">
          {/* Guarded like the molecule dialog: with no SMILES the depiction sits
              on its loading skeleton forever, so an empty field reads as "still
              working" instead of "waiting for you". */}
          {trimmed ? (
            <ReactionDepiction reactionSmiles={trimmed} width={260} height={140} />
          ) : (
            <p className="text-sm text-muted-foreground">Enter reaction SMILES to preview</p>
          )}
        </div>
      </div>
    </EmbedDialogFrame>
  );
}
