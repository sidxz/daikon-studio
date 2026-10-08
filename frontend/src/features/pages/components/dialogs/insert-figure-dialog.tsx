"use client";

import type { Editor } from "@tiptap/react";
import { useState } from "react";
import { toast } from "sonner";

import { uploadBlob } from "@/features/pages/lib/blobs";
import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";

import type { EmbedEditRequest } from "../embed-toolbar";
import { EmbedDialogFrame } from "./embed-dialog-frame";

/** Upload-an-image-and-insert-a-figureImage-node dialog. Controlled (`open`/
 *  `onOpenChange`) so the caller owns when it's shown — page-editor.tsx opens it
 *  from the slash menu, the Insert menu, or an existing node's Edit button. */
export function InsertFigureDialog({
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
  const [file, setFile] = useState<File | null>(null);
  // Seeded at mount from the node being edited (see EmbedDialogFrame: every open
  // is a fresh mount). The file input is the exception — a browser won't let us
  // prefill one, so when editing it means "replace the image" and staying empty
  // keeps the current one.
  const [caption, setCaption] = useState((editing?.attrs.caption as string) ?? "");
  const [alt, setAlt] = useState((editing?.attrs.alt as string) ?? "");
  const [uploading, setUploading] = useState(false);

  async function confirm() {
    if (!file && !editing) return;
    setUploading(true);
    try {
      // Only upload when a new file was picked; editing just the caption/alt
      // leaves the existing blob (and its key) alone.
      const uploaded = file ? await uploadBlob(file) : null;
      if (editor.isDestroyed) return;
      const text = { caption: caption.trim() || null, alt };
      if (editing) {
        editing.onUpdate(
          uploaded ? { blobKey: uploaded.sha256, mime: uploaded.mime, ...text } : text,
        );
      } else {
        editor
          .chain()
          .focus()
          .insertContent({
            type: "figureImage",
            attrs: { blobKey: uploaded!.sha256, mime: uploaded!.mime, ...text },
          })
          .run();
      }
      onOpenChange(false);
    } catch {
      // Keep the dialog open (with the chosen file still selected) so the user
      // can just retry instead of re-picking the file.
      toast.error("Couldn't upload the figure. Try again.");
      setUploading(false);
    }
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit figure" : "Insert figure"}
      description={
        editing
          ? "Update the caption or alt text, or pick a new image to replace this one."
          : "Upload an image and add an optional caption."
      }
      confirmLabel={editing ? "Save changes" : "Insert figure"}
      busyLabel="Uploading…"
      busy={uploading}
      canConfirm={!!file || !!editing}
      onConfirm={() => void confirm()}
    >
      {/* Fixed height so picking a file or typing a caption never resizes the
          dialog (no layout jump) — tall enough that the three fields fit
          without scrolling. */}
      <div className="flex h-60 flex-col gap-4 overflow-y-auto">
        <Field>
          <FieldLabel htmlFor="insert-figure-file">
            {editing ? "Replace image (optional)" : "Image file"}
          </FieldLabel>
          <Input
            id="insert-figure-file"
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </Field>

        <Field>
          <FieldLabel htmlFor="insert-figure-caption">Caption (optional)</FieldLabel>
          <Input
            id="insert-figure-caption"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="What this figure shows"
          />
        </Field>

        <Field>
          <FieldLabel htmlFor="insert-figure-alt">Alt text (optional)</FieldLabel>
          <Input
            id="insert-figure-alt"
            value={alt}
            onChange={(e) => setAlt(e.target.value)}
            placeholder="Describes the image for screen readers"
          />
        </Field>
      </div>
    </EmbedDialogFrame>
  );
}
