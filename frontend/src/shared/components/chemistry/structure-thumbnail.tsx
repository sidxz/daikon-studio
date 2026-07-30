"use client";

import { getRDKit } from "@/shared/lib/rdkit/rdkit-loader";
import { cn } from "@/shared/lib/utils";
import { memo, useEffect, useState } from "react";

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
  }, [smiles, size]);

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
    <img
      src={svgUrl}
      alt={smiles}
      width={size}
      height={size}
      // The SVG is drawn with black bonds; inverting in dark mode is cheaper and
      // sharper than re-rendering with a themed palette.
      className={cn("dark:invert", className)}
    />
  );
});
