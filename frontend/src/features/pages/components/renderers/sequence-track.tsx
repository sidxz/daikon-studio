"use client";

import { cn } from "@/shared/lib/utils";

export type SequenceFeature = {
  start: number; // 1-based, inclusive — matches UniProt/genomic feature convention
  end: number; // 1-based, inclusive
  type?: string;
  label?: string;
};

const RESIDUES_PER_ROW = 60;

// ponytail: flat per-base tint for DNA/RNA only — protein residues render as
// plain monospace text. A hydrophobicity/charge color scheme is more than a
// "modest" static track needs; add it (or swap this whole component) if a
// richer view is ever warranted — see the Nightingale note below.
const BASE_TINT: Record<string, string> = {
  A: "text-emerald-600 dark:text-emerald-400",
  C: "text-sky-600 dark:text-sky-400",
  G: "text-amber-600 dark:text-amber-400",
  T: "text-rose-600 dark:text-rose-400",
  U: "text-rose-600 dark:text-rose-400",
};

// Cycled by feature index so adjacent/overlapping annotations stay visually
// distinct without requiring a caller-supplied color per feature.
const FEATURE_TINTS = [
  "bg-amber-200/70 dark:bg-amber-900/60",
  "bg-sky-200/70 dark:bg-sky-900/60",
  "bg-rose-200/70 dark:bg-rose-900/60",
  "bg-violet-200/70 dark:bg-violet-900/60",
  "bg-emerald-200/70 dark:bg-emerald-900/60",
];

/**
 * Static, presentational sequence track: monospace residues chunked into
 * fixed-width rows, with feature spans drawn as colored underlays. Pure
 * SVG/CSS — no external lib. A lab-notebook embed needs a legible strip of a
 * pasted sequence, not a full genome-browser widget.
 *
 * ponytail: static track; reach for Nightingale (or similar) later if rich
 * multi-track/zoomable feature rendering is ever needed — this component is
 * the whole upgrade seam (NodeView renders only through it).
 */
export function SequenceTrack({
  seqType,
  sequence,
  features = [],
}: {
  seqType: "protein" | "dna";
  sequence: string;
  features?: SequenceFeature[];
}) {
  const rows: string[] = [];
  for (let i = 0; i < sequence.length; i += RESIDUES_PER_ROW) {
    rows.push(sequence.slice(i, i + RESIDUES_PER_ROW));
  }

  return (
    <div
      data-sequence-track=""
      data-seq-type={seqType}
      className="overflow-x-auto rounded-md border border-border bg-muted/20 p-3 font-mono text-sm leading-6"
    >
      {rows.map((row, rowIndex) => {
        const rowStart = rowIndex * RESIDUES_PER_ROW; // 0-based offset of this row's first residue
        return (
          <div key={rowIndex} data-sequence-row="" className="flex whitespace-pre">
            <span
              aria-hidden
              className="mr-3 w-10 shrink-0 select-none text-right text-xs text-muted-foreground"
            >
              {rowStart + 1}
            </span>
            {row.split("").map((residue, i) => {
              const position = rowStart + i + 1; // 1-based position in the full sequence
              // ponytail: first match wins — overlapping features tint by the
              // earlier one. Index (not just the feature) so the tint can cycle.
              const featureIndex = features.findIndex(
                (f) => position >= f.start && position <= f.end,
              );
              const feature = features[featureIndex];
              const hoverText = feature ? (feature.label ?? feature.type ?? "feature") : undefined;
              return (
                <span
                  key={i}
                  data-residue=""
                  data-in-feature={feature ? "true" : undefined}
                  title={hoverText}
                  aria-label={hoverText ? `${residue}, ${hoverText}` : undefined}
                  className={cn(
                    "inline-block w-[1ch] text-center",
                    seqType === "dna" && BASE_TINT[residue.toUpperCase()],
                    feature && FEATURE_TINTS[featureIndex % FEATURE_TINTS.length],
                  )}
                >
                  {residue}
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
