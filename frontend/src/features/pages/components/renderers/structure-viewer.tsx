"use client";

import { useEffect, useState } from "react";

import { loadMolstar } from "@/features/pages/lib/molstar/loader";
import { type Rgb, readBackgroundRgb } from "@/features/pages/lib/molstar/theme";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { cn } from "@/shared/lib/utils";

import type { EmbedWidth } from "../embed-toolbar";

// React 19 moved the JSX namespace under React.JSX. `pdbe-molstar` is a
// runtime-registered web component (lib/molstar/loader.ts), not a component
// TypeScript knows about, so it needs its own IntrinsicElements entry — kept
// minimal, just the attributes this file sets (mirrors the inline
// `declare module "@rdkit/rdkit"` augmentation in reaction-depiction.tsx).
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace -- ambient global augmentation of React.JSX requires the `namespace` keyword, no ES2015-module equivalent
  namespace React {
    // eslint-disable-next-line @typescript-eslint/no-namespace -- see above
    namespace JSX {
      interface IntrinsicElements {
        "pdbe-molstar": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
          "molecule-id"?: string;
          "hide-controls"?: string;
          "bg-color-r"?: number;
          "bg-color-g"?: number;
          "bg-color-b"?: number;
          "pdbe-link"?: string;
          "hide-expand-icon"?: string;
          "hide-selection-icon"?: string;
          "hide-animation-icon"?: string;
          "hide-control-toggle-icon"?: string;
          "hide-control-info-icon"?: string;
        };
      }
    }
  }
}

type Status = "loading" | "loaded" | "error";

/** A 3D canvas needs an explicit height — it has no intrinsic one to lay out
 *  from. Horizontal size comes from the container (the node view's <figure>
 *  carries EMBED_WIDTH_CLASS), so this only maps the size preset to a height.
 *  Shared with the loading/error placeholders so every state of a given size
 *  occupies the same box (no layout jump once Mol* paints). */
const HEIGHT_CLASS: Record<EmbedWidth, string> = {
  full: "h-[360px]",
  half: "h-[300px]",
  fit: "h-[220px]",
};

/**
 * Lazy Mol* 3D structure viewer. Loads the pdbe-molstar CDN bundle then
 * renders its web component directly as JSX — React 19 passes unrecognized
 * dashed-name props through as element attributes, so (unlike ProtCellar's
 * card, written before that support) no imperative
 * `createElement`/`setAttribute` dance is needed.
 *
 * v1 only wires `source: "pdb"` — Mol* fetches coordinates straight from RCSB
 * by ID, so it's self-contained (no backend). `"protcellar"` is an
 * unused-by-v1 seam, same shape as chemStructure's chembl/chemcellar sources.
 */
export function StructureViewer({
  source,
  sourceId,
  width = "full",
}: {
  source: "pdb" | "protcellar";
  sourceId: string;
  width?: EmbedWidth;
}) {
  const [status, setStatus] = useState<Status>("loading");
  // Lazy initial read (client-only; `bg` is never part of the SSR output because
  // status starts at "loading"), then kept in step by watching <html> itself.
  // next-themes' resolvedTheme is deliberately NOT the trigger: it updates a
  // render *before* the data-theme attribute lands, so reading off it hands back
  // the outgoing theme's colour. The attribute mutation is the real signal.
  const [bg, setBg] = useState<Rgb>(readBackgroundRgb);

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const next = readBackgroundRgb();
      // Same-value guard: <html> also churns on class/style, and a fresh object
      // every time would remount the viewer (and re-fetch from RCSB) for nothing.
      setBg((prev) => (prev.r === next.r && prev.g === next.g && prev.b === next.b ? prev : next));
    });
    observer.observe(document.documentElement, { attributes: true });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (source !== "pdb" || !sourceId) return;
    let cancelled = false;
    void (async () => {
      try {
        await loadMolstar();
        if (!cancelled) setStatus("loaded");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [source, sourceId]);

  const box = cn(HEIGHT_CLASS[width], "w-full");
  const placeholder = cn(
    box,
    "flex items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground",
  );

  if (source !== "pdb") {
    return <div className={placeholder}>ProtCellar structures aren&apos;t available yet.</div>;
  }
  if (status === "error") {
    return <div className={placeholder}>Couldn&apos;t load the 3D viewer.</div>;
  }
  if (status !== "loaded") {
    // ponytail: a thumbnail-first paint (snapshot.thumbnailBlobKey) is a
    // nice-to-have per the task brief; no capture pipeline exists yet to
    // populate that key, so a plain skeleton covers v1.
    return <Skeleton className={cn(box, "rounded-md")} />;
  }

  return (
    <div
      data-structure-embed=""
      className={cn(box, "relative overflow-hidden rounded-md bg-background")}
    >
      {/* Mol* lays its canvas out as position:absolute inset:0 — fill this
          (position:relative) host so it never escapes to the viewport.
          Keyed by the background colour: bg-color-* is read when the viewer
          initialises, so a remount is what actually repaints the canvas.
          ponytail: remount re-fetches the entry from RCSB — theme toggling is
          rare enough not to warrant reaching into Mol*'s imperative API. */}
      {/* pdbe-link is off because its logo points at an EBI image URL that 404s,
          so it renders as a broken-image box. The viewport controls are pared down
          to the one that matters in a page embed — expand — since a document reader
          wants the figure, not a workbench. The rest (selection mode, animation,
          controls panel, settings) are hidden for good: expanding reuses this same
          DOM, so they stay hidden there too; rotate/zoom by mouse still work in both.
          What these attributes can't reach (reset zoom, screenshot) plus the
          auto-hide and theming are handled in globals.css, which is also where the
          library's lack of theming hooks is explained. */}
      <pdbe-molstar
        // sourceId is in the key because Mol* reads molecule-id only when it
        // initialises: editing a node's PDB entry swaps the attribute, which on
        // its own leaves the old structure on screen (the caption updates and the
        // canvas doesn't). Remounting is what actually reloads it.
        key={`${sourceId}-${bg.r}-${bg.g}-${bg.b}-${width}`}
        molecule-id={sourceId.toLowerCase()}
        hide-controls="true"
        pdbe-link="false"
        hide-animation-icon="true"
        hide-selection-icon="true"
        hide-control-toggle-icon="true"
        hide-control-info-icon="true"
        bg-color-r={bg.r}
        bg-color-g={bg.g}
        bg-color-b={bg.b}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />
    </div>
  );
}
