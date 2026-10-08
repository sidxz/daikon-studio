import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { ChartView } from "../chart-view";

/**
 * An embedded chart. An atom node — edited only through the NodeView/insert
 * dialog. `raw` holds the author's pasted CSV/TSV verbatim rather than parsed
 * rows: it is the single source of truth the edit dialog reseeds from, and it
 * keeps a 200-row plot at a few KB inside the page body.
 *
 * Like every other page embed this lives entirely in the ProseMirror doc, so
 * save/history/restore work with no backend change.
 */
export const Chart = Node.create({
  name: "chart",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      kind: { default: "bar" }, // "bar" | "line" | "scatter" | "radar" | "pie"
      raw: { default: "" },
      // null means "resolve from the data" (see resolveColumns); an empty
      // series array would instead mean the author deselected every column.
      x: { default: null },
      series: { default: null },
      options: { default: {} },
      caption: { default: null },
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-chart]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-chart": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(ChartView);
  },
});
