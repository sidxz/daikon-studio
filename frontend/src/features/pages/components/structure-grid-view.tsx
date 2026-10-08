"use client";

import type { NodeViewProps } from "@tiptap/react";

import type { StructureItem } from "@/features/pages/lib/molecules";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import { MoleculeDepiction } from "./renderers/molecule-depiction";

export const SOURCE_LABEL: Record<string, string> = {
  smiles: "SMILES",
  chembl: "ChEMBL",
  chemcellar: "ChemCellar",
  collection: "a ChemCellar collection",
};

/**
 * The quiet line under the grid. `capturedAt` earns its place here more than on
 * other embeds: a ChemCellar collection is a living set, so a grid that no
 * longer matches it should look stale rather than be silently wrong.
 */
export function provenanceLine(source: string, count: number, capturedAt: string | null): string {
  const noun = count === 1 ? "structure" : "structures";
  const head = `${count} ${noun} from ${SOURCE_LABEL[source] ?? source}`;

  if (!capturedAt) return head;
  const when = new Date(capturedAt);
  // An unparseable stored value must not render as "Invalid Date".
  if (Number.isNaN(when.getTime())) return head;
  const stamp = when.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  return `${head} · captured ${stamp}`;
}

function StructureCard({ item }: { item: StructureItem }) {
  return (
    // h-full so a two-line name can't leave its row ragged — see the app-wide
    // equal-cards rule.
    <li className="flex h-full flex-col overflow-hidden rounded-md border border-border">
      <div className="flex flex-1 items-center justify-center bg-background p-2">
        {item.smiles ? (
          <MoleculeDepiction smiles={item.smiles} className="h-auto w-full" />
        ) : (
          // Resolved, but there is nothing to draw: a ChEMBL biologic or a
          // ChemCellar undisclosed compound. Deliberately NOT routed through
          // MoleculeDepiction, whose empty-input path claims "Invalid
          // structure" — that would assert the SMILES is malformed when in
          // fact none exists.
          <span className="py-8 text-xs text-muted-foreground">No structure</span>
        )}
      </div>
      <div className="border-t border-border px-2 py-1.5">
        {item.id ? <p className="text-xs text-muted-foreground">{item.id}</p> : null}
        <p className="text-sm text-foreground">{item.name}</p>
      </div>
    </li>
  );
}

export function StructureGridView(props: NodeViewProps) {
  const { items, source, caption, capturedAt, width } = props.node.attrs as {
    items: StructureItem[];
    source: string;
    caption: string | null;
    capturedAt: string | null;
    width: EmbedWidth | null;
  };
  const list = Array.isArray(items) ? items : [];

  return (
    <EmbedFigure {...props} kind="structureGrid" width={width ?? "full"}>
      <ul className="grid list-none grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3 p-0">
        {list.map((item, i) => (
          // SMILES are not unique — a grid may legitimately hold the same
          // structure twice under different names — so the index is part of the
          // key. The list is frozen and never reordered in place, so this is
          // stable.
          <StructureCard key={`${item.id ?? item.smiles}-${i}`} item={item} />
        ))}
      </ul>
      <figcaption className="mt-1 text-sm">
        {caption ? <span className="text-foreground">{caption}</span> : null}
        <span className="block text-muted-foreground">
          {provenanceLine(source, list.length, capturedAt)}
        </span>
      </figcaption>
    </EmbedFigure>
  );
}
