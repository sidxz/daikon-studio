"use client";

import { StructureViewer } from "@/features/pages/components/renderers/structure-viewer";
import { useCallback, useEffect, useMemo, useState } from "react";

/**
 * A docked pose drawn inside the receptor it was docked into.
 *
 * A docking score is the least reliable number a structure yields, so the picture is not
 * decoration: it is where a chemist decides whether the pose is chemically sensible at
 * all. Does it reach the residue the site is named for, is the charged group paired or
 * left desolvated, is the ligand buried or hanging into solvent. None of that is in a
 * column of affinities, and a run whose poses are never looked at has only been scored.
 */

/** PDBQT keeps PDB's fixed columns up to the coordinates and adds its own after them. */
const PDB_COLUMNS = 54;

/**
 * One PDB file holding the receptor and one pose.
 *
 * Mol* loads a single coordinate file, so the two are concatenated rather than layered.
 * The pose's own records are rewritten as HETATM of a residue named LIG: left as ATOM it
 * would be read as part of the chain, bonded to whatever it sits next to, and drawn as
 * protein. Everything past the coordinate columns is dropped, because the partial charge
 * and the AutoDock atom type that live there are not PDB and some parsers choke on them.
 */
export function combinedPdb(receptor: string, pose: string): string {
  const kept = receptor
    .split("\n")
    .filter((line) => line.startsWith("ATOM") || line.startsWith("HETATM"));
  const ligand = pose
    .split("\n")
    .filter((line) => line.startsWith("ATOM") || line.startsWith("HETATM"))
    .map((line, index) => {
      const body = line.slice(0, PDB_COLUMNS).padEnd(PDB_COLUMNS, " ");
      const serial = String(index + 1).padStart(5, " ");
      // Chain Z, residue 900: a label no receptor in this product uses, so the pose can
      // always be selected on its own.
      return `HETATM${serial} ${body.slice(12, 17)}LIG Z 900    ${body.slice(30, PDB_COLUMNS)}`;
    });
  return [...kept, ...ligand, "END"].join("\n");
}

/** The chain and residue the pose is written as, so it can always be selected on its own. */
const LIGAND_CHAIN = "Z";
const LIGAND_RESIDUE = 900;
/** How close a residue has to come to the ligand to count as lining it, in angstroms. */
const CONTACT = 4.5;
/** Amber for the ligand, teal for the wall it packs against. Neither is a CPK element
 *  colour, so the eye separates "what I docked" from "what it is sitting in". */
const LIGAND_COLOUR = { r: 245, g: 158, b: 11 };
const LINING_COLOUR = { r: 13, g: 148, b: 136 };

type Atom = { chain: string; resseq: number; x: number; y: number; z: number };

function atoms(text: string, ligand: boolean): Atom[] {
  const out: Atom[] = [];
  for (const line of text.split("\n")) {
    const isHet = line.startsWith("HETATM");
    if (!line.startsWith("ATOM") && !isHet) continue;
    if (isHet !== ligand) continue;
    out.push({
      chain: line[21],
      resseq: Number.parseInt(line.slice(22, 26), 10),
      x: Number.parseFloat(line.slice(30, 38)),
      y: Number.parseFloat(line.slice(38, 46)),
      z: Number.parseFloat(line.slice(46, 54)),
    });
  }
  return out;
}

/**
 * The residues the pose actually touches.
 *
 * Taken from the pose rather than from the pocket that was approved, and deliberately:
 * the box is wider than the site, so a compound can land somewhere the pocket's residue
 * list never mentions. Showing the approved lining would then paint the place the pose
 * was supposed to go instead of the place it went, which is the one thing a reviewer
 * must not be shown.
 */
export function liningResidues(
  text: string,
  cutoff = CONTACT,
): { chain: string; resseq: number }[] {
  const ligand = atoms(text, true);
  if (ligand.length === 0) return [];
  const near = new Map<string, { chain: string; resseq: number }>();
  const square = cutoff * cutoff;
  for (const atom of atoms(text, false)) {
    for (const l of ligand) {
      const dx = atom.x - l.x;
      const dy = atom.y - l.y;
      const dz = atom.z - l.z;
      if (dx * dx + dy * dy + dz * dz <= square) {
        near.set(`${atom.chain}:${atom.resseq}`, { chain: atom.chain, resseq: atom.resseq });
        break;
      }
    }
  }
  return [...near.values()].sort((a, b) =>
    a.chain === b.chain ? a.resseq - b.resseq : a.chain.localeCompare(b.chain),
  );
}

type ViewerInstance = {
  events: { loadComplete: { subscribe: (cb: () => void) => { unsubscribe: () => void } } };
  visual: {
    select: (params: { data: unknown[] }) => Promise<void>;
    focus: (data: unknown[]) => Promise<void>;
  };
};
type ViewerElement = HTMLElement & { viewerInstance?: ViewerInstance };

export function PoseViewer({
  receptor,
  pose,
  name = "pose",
}: {
  receptor: string;
  pose: string;
  /** Used for the downloaded file's name, so a folder of them stays legible. */
  name?: string;
}) {
  const text = useMemo(() => combinedPdb(receptor, pose), [receptor, pose]);
  const [url, setUrl] = useState<string | null>(null);
  const [viewer, setViewer] = useState<ViewerInstance | null>(null);
  const [loaded, setLoaded] = useState(false);
  const lining = useMemo(() => liningResidues(text), [text]);

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

  // Cartoon over a whole protein is a picture of a fold, not of a binding mode. The
  // residues the pose touches go in as sticks, the pose itself is coloured apart from
  // them, and the camera lands on the pose -- which is the only part anyone is judging.
  useEffect(() => {
    if (!viewer || !loaded) return;
    const ligand = [
      {
        auth_asym_id: LIGAND_CHAIN,
        start_auth_residue_number: LIGAND_RESIDUE,
        end_auth_residue_number: LIGAND_RESIDUE,
        color: LIGAND_COLOUR,
        representation: "ball-and-stick",
        representationColor: LIGAND_COLOUR,
      },
    ];
    const wall = lining.map((residue) => ({
      auth_asym_id: residue.chain,
      start_auth_residue_number: residue.resseq,
      end_auth_residue_number: residue.resseq,
      color: LINING_COLOUR,
      sideChain: true,
      representation: "ball-and-stick",
      representationColor: LINING_COLOUR,
    }));
    void (async () => {
      await viewer.visual.select({ data: [...wall, ...ligand] });
      await viewer.visual.focus(ligand);
    })();
  }, [viewer, loaded, lining]);

  useEffect(() => {
    // A blob URL rather than a data URL: a receptor and a pose together run past the
    // length some browsers accept in a URL, and Mol* fetches whatever it is handed.
    const made = URL.createObjectURL(new Blob([text], { type: "chemical/x-pdb" }));
    setUrl(made);
    return () => URL.revokeObjectURL(made);
  }, [text]);

  if (!url) return null;
  return (
    <div className="space-y-1">
      <StructureViewer source="url" sourceId={url} width="full" elementRef={attach} />
      {/* Looking is where a pose is judged; opening it somewhere else is where it is
          worked on. The same file serves both, so the link is the one already made. */}
      <a
        href={url}
        download={`${name}.pdb`}
        className="text-xs text-muted-foreground underline underline-offset-2"
      >
        Download this pose with its receptor (.pdb)
      </a>
      <p className="text-xs text-muted-foreground">
        {lining.length} residues within {CONTACT} A of the pose, shown as sticks.
      </p>
    </div>
  );
}
