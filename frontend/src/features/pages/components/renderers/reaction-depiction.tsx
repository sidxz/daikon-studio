"use client";

import { useTheme } from "next-themes";
import { memo, useEffect, useState } from "react";

import { getRDKit } from "@/shared/lib/rdkit/rdkit-loader";
import { cn } from "@/shared/lib/utils";

import { drawingOptions } from "./molecule-depiction";

// `get_rxn`/`JSReaction` are real, working methods in the installed
// 2025.3.4-1.0.0 MinimalLib build (confirmed both by `strings` on
// RDKit_minimal.wasm — `get_rxn`, `JSReaction`, `run_reactants`, `get_svg` all
// present — and by a runtime probe: `rdkit.get_rxn(rxnSmiles).get_svg(w, h)`
// returns a real multi-fragment reaction SVG, and `get_svg_with_highlights`
// honours the same JSON drawing options as a mol's). The bundled `@rdkit/rdkit`
// `.d.ts` just never got updated to declare them on `RDKitModule`, so this
// augments it rather than casting through `unknown` at every call site.
declare module "@rdkit/rdkit" {
  export interface JSReaction {
    get_svg(width?: number, height?: number): string;
    get_svg_with_highlights(details: string): string;
    delete(): void;
  }

  export interface RDKitModule {
    /** Throws (not an `is_valid()` gate like `get_mol`) on malformed input. */
    get_rxn(input: string): JSReaction;
  }
}

type Depiction = { url: string } | { error: true };

/**
 * Renders a reaction scheme from reaction SMILES using RDKit-JS (WASM).
 * Same SVG-blob-URL + cleanup discipline, drawing options and wait for the
 * theme as `<MoleculeDepiction>` — see there for the "no synchronous setState
 * in the effect body" rationale.
 */
function ReactionDepictionInner({
  reactionSmiles,
  width = 480,
  height = 200,
  className,
}: {
  reactionSmiles: string;
  width?: number;
  height?: number;
  className?: string;
}) {
  const [depiction, setDepiction] = useState<Depiction | null>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    if (!reactionSmiles || !resolvedTheme) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    void (async () => {
      try {
        const rdkit = await getRDKit();
        if (cancelled) return;
        const rxn = rdkit.get_rxn(reactionSmiles);
        let svg: string;
        try {
          svg = rxn.get_svg_with_highlights(
            JSON.stringify({ width, height, ...drawingOptions(resolvedTheme) }),
          );
        } finally {
          rxn.delete();
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
  }, [reactionSmiles, width, height, resolvedTheme]);

  const current = reactionSmiles && resolvedTheme ? depiction : null;

  if (!current) {
    // See molecule-depiction.tsx: dynamic numeric sizing, so the underlying
    // `.skeleton` class is applied directly rather than via <Skeleton>.
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
        Invalid reaction
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element -- blob: URL, see molecule-depiction.tsx
    <img
      src={current.url}
      alt={`Reaction: ${reactionSmiles}`}
      width={width}
      height={height}
      className={className}
    />
  );
}

export const ReactionDepiction = memo(ReactionDepictionInner);
