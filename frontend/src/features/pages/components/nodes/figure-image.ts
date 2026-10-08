import { Node, ReactNodeViewRenderer, mergeAttributes } from "@tiptap/react";

import { FigureImageView } from "../figure-image-view";

/**
 * An uploaded figure. An atom node — edited only through the NodeView/insert
 * dialog, never by typing into it — wrapping a content-addressed blob. The doc
 * JSON only ever carries `blobKey` (the blob's sha256), never bytes; the
 * NodeView resolves it to pixels via `useBlobObjectUrl` (lib/pages/blobs.ts).
 */
export const FigureImage = Node.create({
  name: "figureImage",
  group: "block",
  atom: true,

  addAttributes() {
    return {
      blobKey: { default: null }, // sha256
      width: { default: "full" }, // "full" | "half" | "fit" — see EMBED_WIDTH_CLASS
      mime: { default: null },
      caption: { default: null },
      alt: { default: "" },
    };
  },

  parseHTML() {
    return [{ tag: "figure[data-figure-image]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["figure", mergeAttributes({ "data-figure-image": "" }, HTMLAttributes)];
  },

  addNodeView() {
    return ReactNodeViewRenderer(FigureImageView);
  },
});
