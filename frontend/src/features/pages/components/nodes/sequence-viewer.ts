import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { SequenceViewerView } from "../sequence-viewer-view";

/**
 * A lightweight annotated-sequence embed. An atom node — edited only through
 * the NodeView/insert dialog. `sequence` is self-contained (raw pasted text),
 * so v1 needs no backend and no external lib. `features` is `[]` until a
 * later annotation pass; `source`/`sourceId` are an unused-by-v1 seam for a
 * later cellar-search refresh (same shape as chemStructure's
 * chembl/chemcellar seam).
 */
export const SequenceViewer = Node.create({
  name: "sequenceViewer",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      seqType: { default: "protein" }, // "protein" | "dna"
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
      sequence: { default: "" },
      features: { default: [] }, // { start, end, type, label }[]
      source: { default: null },
      sourceId: { default: null },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-sequence-viewer]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-sequence-viewer": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(SequenceViewerView);
  },
});
