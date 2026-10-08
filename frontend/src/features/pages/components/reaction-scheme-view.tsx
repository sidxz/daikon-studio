"use client";

import type { NodeViewProps } from "@tiptap/react";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import { ReactionDepiction } from "./renderers/reaction-depiction";

export function ReactionSchemeView(props: NodeViewProps) {
  const { reactionSmiles, title, conditions, width } = props.node.attrs as {
    reactionSmiles: string;
    title: string | null;
    conditions: string | null;
    width: EmbedWidth | null;
  };
  const size = width ?? "full";

  return (
    <EmbedFigure {...props} kind="reaction" width={size}>
      {title ? <div className="text-sm font-medium">{title}</div> : null}
      {/* SVG depiction — scales losslessly to whatever width the figure gets. */}
      <ReactionDepiction reactionSmiles={reactionSmiles} className="h-auto w-full" />
      {conditions ? (
        <figcaption className="mt-1 text-sm text-muted-foreground">{conditions}</figcaption>
      ) : null}
    </EmbedFigure>
  );
}
