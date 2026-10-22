"use client";

import { useCallback, useEffect, useState } from "react";

import { loadMolstar } from "@/features/pages/lib/molstar/loader";
import { type Rgb, readBackgroundRgb } from "@/features/pages/lib/molstar/theme";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { cn } from "@/shared/lib/utils";

export type PocketResidue = { chain: string; resname: string; resseq: number };

/** Magenta: none of the CPK element colours the ligands are drawn in, so the
 *  lining reads as lining and not as part of the cofactor. */
const LINING: Rgb = { r: 192, g: 38, b: 211 };
/** Every other candidate, in one muted colour. Whether the neighbouring patches
 *  abut the selected one is the question worth answering here — fpocket splits
 *  an extended site into several entries, and four fragments of one cleft look
 *  exactly like four separate sites in a ranked table. */
const OTHER: Rgb = { r: 234, g: 179, b: 8 };

/**
 * One Mol* selection entry per lining residue, in author numbering.
 *
 * fpocket read author numbering off the prepared file (PDBFixer keeps the ids),
 * and PDBe serves the same numbers as `auth_seq_id`, so the two line up without
 * an offset. A single residue goes in as a one-long range because that is the
 * form pdbe-molstar definitely resolves; its bare `auth_residue_number` path is
 * entangled with insertion codes. Getting this mapping wrong paints the wrong
 * residues and says nothing, which is the bug class this viewer exists to
 * catch, so it is a pure function with a test.
 */
export function selectionData(residues: PocketResidue[], color: Rgb = LINING) {
  return residues.map((residue) => ({
    auth_asym_id: residue.chain,
    start_auth_residue_number: residue.resseq,
    end_auth_residue_number: residue.resseq,
    color,
    sideChain: true,
    // The wall of the cavity, not just a coloured ribbon: the solvent-excluded
    // surface of the lining is the face a ligand would pack against, and
    // whether it is a cup or a dent on the outside is the whole question.
    representation: "molecular-surface",
    representationColor: color,
  }));
}

type ViewerInstance = {
  events: { loadComplete: { subscribe: (cb: () => void) => { unsubscribe: () => void } } };
  visual: {
    select: (params: { data: unknown[] }) => Promise<void>;
    focus: (data: unknown[]) => Promise<void>;
  };
};

type ViewerElement = HTMLElement & { viewerInstance?: ViewerInstance };

/**
 * The deposited entry with the lining residues of one pocket painted and the
 * camera on them. The gate on the pocket step asks which cavity to aim at, and
 * that call is made by looking: whether the candidate is a buried cleft or a
 * dent on an oligomerisation face, and whether it sits against the cofactor,
 * are both unreadable from a volume and a score.
 *
 * `pdbe-molstar` fetches the entry itself, so this needs no coordinates from
 * our API — see the JSX augmentation and the CDN loader behind
 * features/pages/components/renderers/structure-viewer.tsx, which this reuses
 * wholesale (theme watching included).
 */
export function PocketViewer({
  pdbId,
  residues,
  others = [],
  className,
}: {
  pdbId: string;
  residues: PocketResidue[];
  /** Lining of the pockets that were not picked, drawn behind the selected one. */
  others?: PocketResidue[];
  className?: string;
}) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [bg, setBg] = useState<Rgb>(readBackgroundRgb);
  const [viewer, setViewer] = useState<ViewerInstance | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Same signal as the page embed: next-themes' resolvedTheme updates a render
  // before data-theme lands, so the attribute mutation is what to watch.
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const next = readBackgroundRgb();
      setBg((prev) => (prev.r === next.r && prev.g === next.g && prev.b === next.b ? prev : next));
    });
    observer.observe(document.documentElement, { attributes: true });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await loadMolstar();
        if (!cancelled) setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /**
   * A ref callback rather than an effect, because the lifecycle that matters is
   * the element's: it is keyed by entry and background, so a theme toggle hands
   * back a fresh plugin with nothing loaded, and the subscription has to follow
   * the new one. React 19 runs the returned cleanup when that element goes.
   *
   * `viewerInstance` is assigned in the component's `connectedCallback`, which
   * has run by the time React attaches a ref — the element is only rendered
   * once `loadMolstar` has defined it, so it is never an un-upgraded tag.
   *
   * ponytail: `loadComplete` is a plain subject with no replay, so a load that
   * finished before this subscribed would be missed. It cannot: attaching is
   * part of the same commit that inserts the element, and the load is a network
   * round trip away.
   */
  const attach = useCallback((element: HTMLElement | null) => {
    const instance = (element as ViewerElement | null)?.viewerInstance ?? null;
    setViewer(instance);
    const subscription = instance?.events.loadComplete.subscribe(() => setLoaded(true));
    return () => {
      subscription?.unsubscribe();
      setViewer(null);
      setLoaded(false);
    };
  }, []);

  useEffect(() => {
    if (!viewer || !loaded) return;
    const selected = selectionData(residues);
    // The selected pocket goes last so it wins the residues it shares with a
    // neighbour — which is the overlap worth seeing.
    const data = [...selectionData(others, OTHER), ...selected];
    void (async () => {
      await viewer.visual.select({ data });
      if (selected.length > 0) await viewer.visual.focus(selected);
    })();
  }, [viewer, loaded, residues, others]);

  const box = cn("h-[380px] w-full", className);
  if (status === "error") {
    return (
      <div
        className={cn(
          box,
          "flex items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground",
        )}
      >
        Couldn&apos;t load the 3D viewer.
      </div>
    );
  }
  if (status !== "ready") return <Skeleton className={cn(box, "rounded-md")} />;

  return (
    <div
      data-structure-embed=""
      className={cn(box, "relative overflow-hidden rounded-md border bg-background")}
    >
      <pdbe-molstar
        ref={attach}
        key={`${pdbId}-${bg.r}-${bg.g}-${bg.b}`}
        molecule-id={pdbId.toLowerCase()}
        hide-controls="true"
        // Deposited waters are scattered red dots over the whole protein, which
        // is noise in a cavity view; the ones that matter are a later question.
        hide-water="true"
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
