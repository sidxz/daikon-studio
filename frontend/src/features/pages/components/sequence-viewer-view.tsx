"use client";

import type { NodeViewProps } from "@tiptap/react";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import { type SequenceFeature, SequenceTrack } from "./renderers/sequence-track";

export function SequenceViewerView(props: NodeViewProps) {
  const { seqType, sequence, features, width } = props.node.attrs as {
    seqType: "protein" | "dna";
    sequence: string;
    features: SequenceFeature[];
    width: EmbedWidth | null;
  };
  const size = width ?? "full";

  return (
    <EmbedFigure {...props} kind="sequence" width={size}>
      <SequenceTrack seqType={seqType} sequence={sequence} features={features} />
    </EmbedFigure>
  );
}
