"use client";

import type { NodeViewProps } from "@tiptap/react";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import { StructureViewer } from "./renderers/structure-viewer";

export function ProteinStructureView(props: NodeViewProps) {
  const { source, sourceId, snapshot, width } = props.node.attrs as {
    source: "pdb" | "protcellar";
    sourceId: string;
    snapshot: { name?: string; method?: string; resolution?: string } | null;
    width: EmbedWidth | null; // null on nodes saved before the size picker existed
  };
  const size = width ?? "full";

  const captionText = [snapshot?.name, snapshot?.method, snapshot?.resolution]
    .filter(Boolean)
    .join(" · ");

  return (
    <EmbedFigure {...props} kind="protein" width={size}>
      <StructureViewer source={source} sourceId={sourceId} width={size} />
      {captionText ? (
        <figcaption className="mt-1 text-sm text-muted-foreground">{captionText}</figcaption>
      ) : null}
    </EmbedFigure>
  );
}
