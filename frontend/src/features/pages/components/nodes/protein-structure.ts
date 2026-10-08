import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { ProteinStructureView } from "../protein-structure-view";

/**
 * A 3D protein structure embed. An atom node — edited only through the
 * NodeView/insert dialog. `sourceId` (a PDB code) is self-contained — Mol*
 * fetches coordinates straight from RCSB client-side, so v1 needs no backend.
 * `chains`/`style` and the `"protcellar"` source are unused-by-v1 seams for a
 * later ProtCellar-search refresh (same shape as chemStructure's
 * chembl/chemcellar seam).
 */
export const ProteinStructure = Node.create({
  name: "proteinStructure",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      source: { default: "pdb" }, // "pdb" | "protcellar"
      sourceId: { default: null },
      chains: { default: null },
      style: { default: null },
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
      snapshot: { default: null }, // { name?, method?, resolution?, thumbnailBlobKey? } | null
      capturedAt: { default: null },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-protein-structure]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-protein-structure": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(ProteinStructureView);
  },
});
