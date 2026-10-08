import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { StructureGridView } from "../structure-grid-view";

/**
 * A grid of 2D chemical structures. An atom node — edited only through the
 * NodeView/insert dialog.
 *
 * `items` is the frozen result of whichever source filled it (see
 * lib/pages/molecules.ts); the renderer never learns which tab that was beyond
 * the `source` label. Unlike the `chart` node there is no verbatim `raw`
 * alongside the parsed form: a name/SMILES pair round-trips losslessly, so the
 * SMILES tab re-serialises `items` to seed itself on edit and there is exactly
 * one representation that can be right or wrong.
 */
export const StructureGrid = Node.create({
  name: "structureGrid",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      // "smiles" | "chembl" | "chemcellar" | "collection"
      source: { default: "smiles" },
      // [{ smiles, name, id? }] — see StructureItem
      items: { default: [] },
      // ChemCellar collection uuid when source is "collection", else null.
      sourceId: { default: null },
      caption: { default: null },
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
      capturedAt: { default: null },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-structure-grid]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-structure-grid": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(StructureGridView);
  },
});
