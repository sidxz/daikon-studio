"use client";

import { getRDKit } from "@/shared/lib/rdkit/rdkit-loader";
import { cn } from "@/shared/lib/utils";
import { useTheme } from "next-themes";
import { memo, useEffect, useState } from "react";

/**
 * A palette for a dark background.
 *
 * Keyed by atomic number, RDKit's own convention: 0 is the default for
 * anything not listed. Carbon and hydrogen go light so bonds read on a dark
 * surface; the heteroatom colours stay near RDKit's defaults, because a
 * chemist reads oxygen as red and nitrogen as blue and re-teaching that would
 * be a worse crime than a slightly dim red.
 */
const DARK_PALETTE = {
  0: [0.9, 0.9, 0.9],
  1: [0.9, 0.9, 0.9],
  6: [0.9, 0.9, 0.9],
  7: [0.4, 0.55, 1.0],
  8: [1.0, 0.4, 0.4],
  9: [0.4, 0.9, 0.5],
  15: [1.0, 0.6, 0.3],
  16: [0.95, 0.85, 0.3],
  17: [0.4, 0.9, 0.5],
  35: [0.85, 0.55, 0.35],
  53: [0.7, 0.5, 0.9],
};

/**
 * A SMILES rendered as an SVG.
 *
 * Drawn at 2x and scaled down so it stays crisp on a HiDPI screen, handed to
 * an <img> as a blob URL rather than injected as innerHTML, and the RDKit mol
 * is explicitly deleted -- WASM memory is not garbage-collected, so a grid
 * scrolling through thousands of structures leaks without it.
 */
export const StructureThumbnail = memo(function StructureThumbnail({
  smiles,
  size = 96,
  className,
}: {
  smiles: string;
  size?: number;
  className?: string;
}) {
  const [svgUrl, setSvgUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const { resolvedTheme } = useTheme();
  // Undefined pre-hydration and on the server -- falls through to the light
  // palette rather than flashing dark colours before the theme is known.
  const isDark = resolvedTheme === "dark";

  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;

    getRDKit()
      .then((rdkit) => {
        if (cancelled) return;
        const mol = rdkit.get_mol(smiles);
        try {
          if (!mol?.is_valid()) {
            setFailed(true);
            return;
          }
          const svg = mol.get_svg_with_highlights(
            JSON.stringify({
              width: size * 2,
              height: size * 2,
              bondLineWidth: 1.6,
              minFontSize: 13,
              addAtomIndices: false,
              // Transparent, so a structure sits on whatever surface holds it
              // -- a card, a grid row, a dialog -- instead of on a white
              // rectangle that has to be inverted back out in dark mode.
              clearBackground: false,
              ...(isDark ? { atomColourPalette: DARK_PALETTE } : {}),
            }),
          );
          url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
          setSvgUrl(url);
        } finally {
          mol?.delete();
        }
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [smiles, size, isDark]);

  if (failed) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded border border-dashed border-border text-[10px] text-muted-foreground",
          className,
        )}
        style={{ width: size, height: size }}
      >
        no structure
      </div>
    );
  }

  if (!svgUrl) {
    return (
      <div
        className={cn("animate-pulse rounded bg-muted", className)}
        style={{ width: size, height: size }}
      />
    );
  }

  return (
    // A plain <img>: next/image cannot take a blob: URL, and the SVG is
    // generated per-structure at render time rather than served as an asset.
    <img src={svgUrl} alt={smiles} width={size} height={size} className={cn(className)} />
  );
});
