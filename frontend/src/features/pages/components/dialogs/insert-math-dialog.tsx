"use client";

import type { Editor } from "@tiptap/react";
import katex from "katex";
import { useMemo, useState } from "react";

import { Field, FieldLabel } from "@/shared/components/ui/field";
import { Textarea } from "@/shared/components/ui/textarea";

import type { EmbedEditRequest } from "../embed-toolbar";
import { EmbedDialogFrame } from "./embed-dialog-frame";

/** Insert-an-equation dialog. Unlike the science embeds there is no React node
 *  view — @tiptap/extension-mathematics renders KaTeX itself — so `editing`
 *  comes from the extension's onClick (see page-editor.tsx) carrying
 *  latex/mathKind/pos in attrs, and this dialog runs updateInlineMath/
 *  updateBlockMath itself (editing.onUpdate is a stub). pos stays valid while
 *  the dialog is up: it's modal, so the doc can't change before confirm.
 *  Inserts are always block equations; inline math is typed as $$x^2$$
 *  directly in the text. */
export function InsertMathDialog({
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
  const [latex, setLatex] = useState((editing?.attrs.latex as string) ?? "");
  const isBlock = (editing?.attrs.mathKind ?? "block") === "block";
  const trimmed = latex.trim();

  // katex output is sanitized by construction; throwOnError:false renders parse
  // errors as the raw source in red, which doubles as the preview's error state.
  const preview = useMemo(
    () => ({
      __html: trimmed
        ? katex.renderToString(trimmed, { displayMode: isBlock, throwOnError: false })
        : "",
    }),
    [trimmed, isBlock],
  );

  function confirm() {
    if (!trimmed || editor.isDestroyed) return;
    if (editing) {
      const pos = editing.attrs.pos as number;
      const chain = editor.chain().focus();
      (isBlock
        ? chain.updateBlockMath({ latex: trimmed, pos })
        : chain.updateInlineMath({ latex: trimmed, pos })
      ).run();
    } else {
      editor.chain().focus().insertBlockMath({ latex: trimmed }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit equation" : "Insert equation"}
      description={
        editing
          ? "Change this equation's LaTeX."
          : "Write LaTeX; it renders live below. Tip: type $$x^2$$ in the page for inline math."
      }
      confirmLabel={editing ? "Save changes" : "Insert equation"}
      canConfirm={!!trimmed}
      onConfirm={confirm}
    >
      {/* Fixed height so typing never resizes the dialog. */}
      <div className="flex h-64 flex-col gap-4">
        <Field>
          <FieldLabel htmlFor="insert-math-latex">LaTeX</FieldLabel>
          <Textarea
            id="insert-math-latex"
            value={latex}
            onChange={(e) => setLatex(e.target.value)}
            placeholder={"\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}"}
            className="h-24 resize-none font-mono text-sm"
          />
        </Field>
        <div
          className="flex flex-1 items-center justify-center overflow-auto rounded-md border border-border bg-muted/30"
          dangerouslySetInnerHTML={preview}
        />
      </div>
    </EmbedDialogFrame>
  );
}
