import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { ChemStructureView } from "../chem-structure-view";

/**
 * A 2D chemical structure depiction. An atom node — edited only through the
 * NodeView/insert dialog. `smiles` is self-contained: RDKit renders it and
 * computes the `snapshot` (formula/MW) client-side, so v1 needs no backend.
 * `source`/`sourceId` are an unused-by-v1 seam for a later ChEMBL/ChemCellar
 * refresh.
 */
export const ChemStructure = Node.create({
  name: "chemStructure",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      source: { default: "smiles" }, // "smiles" | "chembl" | "chemcellar"
      // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS. Unlike the other embeds
      // this defaults to the small column: a 2D depiction carries far less detail
      // than a 3D structure or a figure, so full width just scales up whitespace.
      width: { default: "fit" },
      smiles: { default: "" },
      sourceId: { default: null },
      snapshot: { default: null }, // { name?, formula, mw } | null
      caption: { default: null }, // author's own text, shown above the provenance line
      capturedAt: { default: null },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-chem-structure]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-chem-structure": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(ChemStructureView);
  },
});
