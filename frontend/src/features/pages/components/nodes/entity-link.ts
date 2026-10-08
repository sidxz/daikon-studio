import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { EntityLinkView } from "../entity-link-view";

/**
 * An inline cross-link chip, inserted via the `#` mention channel (see
 * `onSelect` on `makeMention` in `lib/editor/mention.ts`). `label` freezes at
 * insert time — the doc never re-resolves it, so a later rename elsewhere in
 * the pipeline doesn't retroactively rewrite text already written here.
 *
 * `entityType`/`entityId` are exactly what the backend's `extract_refs`
 * walker (contexts/knowledge/domain/refs.py) keys on to build backlinks
 * (`ref_type: "entity:<type>"`, `ref_id`) — don't rename them.
 */
export const EntityLink = Node.create({
  name: "entityLink",
  group: "inline",
  inline: true,
  atom: true,

  addAttributes() {
    return {
      entityType: { default: null }, // "protein" | "target" | "gene" | ...
      entityId: { default: null },
      label: { default: "" },
    };
  },

  parseHTML() {
    return [{ tag: "span[data-entity-link]" }];
  },

  renderHTML({ node, HTMLAttributes }) {
    return ["span", mergeAttributes({ "data-entity-link": "" }, HTMLAttributes), node.attrs.label];
  },

  addNodeView() {
    return ReactNodeViewRenderer(EntityLinkView);
  },
});
