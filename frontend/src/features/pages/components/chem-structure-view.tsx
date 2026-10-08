"use client";

import type { NodeViewProps } from "@tiptap/react";

import { Badge } from "@/shared/components/ui/badge";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import { MoleculeDepiction } from "./renderers/molecule-depiction";

const SOURCE_LABEL: Record<string, string> = {
  smiles: "SMILES",
  chembl: "ChEMBL",
  chemcellar: "ChemCellar",
};

export function ChemStructureView(props: NodeViewProps) {
  const { smiles, source, snapshot, caption, width } = props.node.attrs as {
    smiles: string;
    source: string;
    snapshot: { name?: string; formula: string; mw: number } | null;
    caption: string | null;
    width: EmbedWidth | null;
  };
  const size = width ?? "fit"; // pre-size-picker molecules rendered inline-block

  const captionText = [
    snapshot?.name,
    snapshot?.formula || null,
    snapshot?.mw ? `MW ${snapshot.mw.toFixed(2)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <EmbedFigure {...props} kind="molecule" width={size}>
      {/* The depiction is an SVG, so filling a wider container scales it
          losslessly rather than blurring. */}
      <MoleculeDepiction smiles={smiles} className="h-auto w-full" />
      {/* The author's caption reads as the figure's label; the derived name /
          formula / MW and the source chip stay below it as quieter provenance. */}
      <figcaption className="mt-1 text-sm">
        {caption ? <span className="text-foreground">{caption}</span> : null}
        <span className="flex items-center gap-2 text-muted-foreground">
          {captionText ? <span>{captionText}</span> : null}
          <Badge variant="outline">{SOURCE_LABEL[source] ?? source}</Badge>
        </span>
      </figcaption>
    </EmbedFigure>
  );
}
