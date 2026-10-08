"use client";

import { useTheme } from "next-themes";
import { memo, useEffect, useState } from "react";

import { getRDKit } from "@/shared/lib/rdkit/rdkit-loader";
import { cn } from "@/shared/lib/utils";

type Depiction = { url: string } | { error: true };

const LIGHT_ON_DARK = [0.886, 0.91, 0.941];

/**
 * RDKit drawing options shared with `<ReactionDepiction>`. Thin bonds, as legacy DAIKON drew them,
 * and no background rect: RDKit's default is opaque white, and transparent lets the surface show
 * through in both themes, as ChemCellar's structure-renderer does. Dark mode gets its own CPK
 * palette rather than a CSS invert, which turns N yellow and O cyan.
 */
export function drawingOptions(resolvedTheme: string) {
  return {
    clearBackground: false,
    bondLineWidth: 1,
    ...(resolvedTheme === "dark" && {
      atomColourPalette: {
        "-1": LIGHT_ON_DARK,
        "0": LIGHT_ON_DARK,
        "1": LIGHT_ON_DARK,
        "6": LIGHT_ON_DARK,
        "7": [0.376, 0.647, 0.98], // N
        "8": [0.973, 0.443, 0.443], // O
        "9": [0.133, 0.827, 0.933], // F
        "15": [0.984, 0.573, 0.235], // P
        "16": [0.98, 0.8, 0.082], // S
        "17": [0.29, 0.871, 0.502], // Cl
        "35": [0.992, 0.729, 0.455], // Br
        "53": [0.655, 0.545, 0.98], // I
      },
      symbolColour: LIGHT_ON_DARK,
      annotationColour: LIGHT_ON_DARK,
    }),
  };
}

/**
 * Renders a 2D chemical structure from SMILES using RDKit-JS (WASM).
 *
 * SVG blob URL (not innerHTML) — the SVG comes from the trusted RDKit WASM
 * library parsing a molecular graph, same recipe as ChemCellar's
 * structure-renderer.tsx. `mol.delete()` always runs (RDKit's WASM heap isn't
 * GC'd). No synchronous setState in the effect body (only after the `await`) —
 * an empty `smiles`, or a theme not yet resolved, short-circuits before any
 * state write, and the render path derives that "nothing to show" case from the
 * current props rather than a separate reset call (mirrors lib/pages/blobs.ts's
 * useBlobObjectUrl). The skeleton holds until the theme is known, so a dark page
 * never flashes a light-palette drawing.
 */
function MoleculeDepictionInner({
  smiles,
  width = 240,
  height = 180,
  className,
}: {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
}) {
  const [depiction, setDepiction] = useState<Depiction | null>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    if (!smiles || !resolvedTheme) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    void (async () => {
      try {
        const rdkit = await getRDKit();
        if (cancelled) return;
        const mol = rdkit.get_mol(smiles);
        if (!mol || !mol.is_valid()) {
          mol?.delete();
          if (!cancelled) setDepiction({ error: true });
          return;
        }
        let svg: string;
        try {
          svg = mol.get_svg_with_highlights(
            JSON.stringify({ width, height, ...drawingOptions(resolvedTheme) }),
          );
        } finally {
          mol.delete();
        }
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
        setDepiction({ url: objectUrl });
      } catch {
        if (!cancelled) setDepiction({ error: true });
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [smiles, width, height, resolvedTheme]);

  const current = smiles && resolvedTheme ? depiction : null;

  if (!current) {
    // The <Skeleton> component only takes a className (sized via Tailwind
    // classes per its own convention); width/height here are dynamic numeric
    // props, so apply its underlying `.skeleton` class directly instead of
    // widening that component's contract for this one dynamic-size caller.
    return <div className={cn("skeleton", className)} style={{ width, height }} aria-hidden />;
  }
  if ("error" in current) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded border border-dashed text-xs text-muted-foreground",
          className,
        )}
        style={{ width, height }}
      >
        Invalid structure
      </div>
    );
  }
  return (
    // src is a client-side blob: object URL — next/image's optimizer fetches
    // from a URL and can't reach a blob: one (see lib/pages/blobs.ts).
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={current.url}
      alt={`Structure: ${smiles}`}
      width={width}
      height={height}
      className={className}
    />
  );
}

export const MoleculeDepiction = memo(MoleculeDepictionInner);

/**
 * Best-effort display snapshot (name is filled in by the caller if known) —
 * not load-bearing, so a computation failure just means a blanker caption.
 *
 * ponytail: RDKit's MinimalLib `get_descriptors()` (verified against the
 * installed 2025.3.4-1.0.0 build via a Node probe) has no Hill-formula key —
 * only `amw`/`exactmw` and physchem descriptors (NumRings, TPSA, …). Rather
 * than hand-roll a Hill-formula counter from `get_molblock()`'s atom block,
 * ship formula empty; revisit if a future MinimalLib build adds one, or if a
 * cellar/ChEMBL refresh (the `source`/`sourceId` seam) ever supplies it.
 */
export async function computeChemSnapshot(
  smiles: string,
): Promise<{ formula: string; mw: number } | null> {
  try {
    const rdkit = await getRDKit();
    const mol = rdkit.get_mol(smiles);
    if (!mol || !mol.is_valid()) {
      mol?.delete();
      return null;
    }
    try {
      const descriptors = JSON.parse(mol.get_descriptors()) as { amw?: number };
      return { formula: "", mw: descriptors.amw ?? 0 };
    } finally {
      mol.delete();
    }
  } catch {
    // Never let a snapshot failure surface as a rejected promise — callers
    // (the insert-molecule dialog) await this mid-confirm with a busy/disabled
    // Cancel button, so an uncaught throw here would strand the dialog open
    // with no way to close it.
    return null;
  }
}
