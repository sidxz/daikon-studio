import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { ReactionSchemeView } from "../reaction-scheme-view";

/**
 * A reaction scheme depiction. An atom node — edited only through the
 * NodeView/insert dialog. `reactionSmiles` is self-contained (RDKit renders it
 * client-side); no `snapshot`/`source` seam yet — v1 has no reaction registry
 * to refresh from (unlike `chemStructure`'s ChEMBL/ChemCellar seam).
 */
export const ReactionScheme = Node.create({
  name: "reactionScheme",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      reactionSmiles: { default: "" },
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
      title: { default: null },
      conditions: { default: null },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-reaction-scheme]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-reaction-scheme": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(ReactionSchemeView);
  },
});
