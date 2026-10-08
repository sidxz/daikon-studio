/**
 * Where a structure grid's contents come from. Every source — a pasted list,
 * ChEMBL, ChemCellar search, a ChemCellar collection — resolves to the same
 * `StructureItem[]`, so the grid renderer never learns which tab produced it.
 *
 * Resolution happens once, in the insert dialog, and the result is frozen into
 * the ProseMirror node (ADR-007's frozen-snapshot rule). Nothing here may be
 * called from a render path.
 */

/** A card in the grid. `smiles` is "" when the source resolved a real compound
 *  that simply has no structure — a ChEMBL biologic, or a ChemCellar compound
 *  registered as undisclosed. That is not an error and must not render as one. */
export type StructureItem = {
  smiles: string;
  name: string;
  /** ChemCellar registration number or ChEMBL id; absent for pasted SMILES. */
  id?: string;
};

/**
 * ponytail: hard cap, not pagination or virtualisation. The binding cost is
 * main-thread: every card is a synchronous RDKit WASM `get_mol` + `get_svg`,
 * and a ChemCellar collection can hold 10 000 members. Raise it by rendering
 * depictions lazily behind an IntersectionObserver — which is what ChemCellar's
 * own card grid does — not by moving this number up on its own.
 */
export const MAX_STRUCTURES = 50;

export function capStructures<T>(items: T[]): { items: T[]; dropped: number } {
  return {
    items: items.slice(0, MAX_STRUCTURES),
    dropped: Math.max(0, items.length - MAX_STRUCTURES),
  };
}

/** Cells named this in a first line mean that line is a header, not a compound. */
const HEADER_KEYS = ["smiles", "structure"];

/**
 * Parse a pasted two-column list into items.
 *
 * Positional by default (name, then SMILES) — that is the order the dialog's
 * placeholder shows. A header row naming a `smiles`/`structure` column is
 * honoured instead, so a spreadsheet paste works in either column order.
 *
 * The header sniff is a keyword match rather than "does this field parse as a
 * structure": deciding it properly would mean an async RDKit call, and the
 * dialog's live preview already makes a wrong guess self-correcting.
 */
export function parseStructureList(raw: string): StructureItem[] {
  const lines = raw.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length === 0) return [];

  // Same rule as lib/pages/parse-table.ts: a tab-delimited paste may contain a
  // comma inside a name, but a comma-delimited one never contains a tab.
  const delimiter = lines.some((l) => l.includes("\t")) ? "\t" : ",";
  const split = (line: string) => line.split(delimiter).map((c) => c.trim());

  const first = split(lines[0]).map((c) => c.toLowerCase());
  const headerAt = first.findIndex((c) => HEADER_KEYS.includes(c));

  let smilesCol = 1;
  let nameCol = 0;
  let body = lines;
  if (headerAt !== -1) {
    smilesCol = headerAt;
    // The name is whichever other column exists; single-column pastes reuse the
    // SMILES as the name below.
    nameCol = first.findIndex((_, i) => i !== headerAt);
    body = lines.slice(1);
  }

  return body.flatMap((line) => {
    const cells = split(line);
    // A one-column paste is a bare SMILES list.
    const smiles = (cells.length === 1 ? cells[0] : cells[smilesCol]) ?? "";
    if (!smiles) return [];
    const name = (nameCol === -1 ? "" : (cells[nameCol] ?? "")) || smiles;
    return [{ name, smiles }];
  });
}

const CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json";

type ChemblMolecule = {
  molecule_chembl_id: string;
  pref_name?: string | null;
  molecule_structures?: { canonical_smiles?: string } | null;
};

/**
 * Resolve a list of ChEMBL ids in ONE request.
 *
 * ChEMBL's data API is public and sends `access-control-allow-origin: *`, so
 * this runs client-side with no backend — the same recipe the single-molecule
 * dialog already uses for a single id. `molecule_chembl_id__in` takes the whole
 * list, so N ids cost one round trip, not N.
 *
 * Caps and de-duplicates before building the URL: this function owns the URL,
 * so it owns not making a 500-id one.
 */
export async function fetchChemblStructures(
  ids: string[],
): Promise<{ items: StructureItem[]; missing: string[]; dropped: number }> {
  const unique = [...new Set(ids.map((i) => i.trim().toUpperCase()).filter(Boolean))];
  const { items: wanted, dropped } = capStructures(unique);
  if (wanted.length === 0) return { items: [], missing: [], dropped };

  // The default page size is 20 — without an explicit limit a longer list
  // silently comes back truncated and every id past the 20th reads as missing.
  const res = await fetch(
    `${CHEMBL_URL}?molecule_chembl_id__in=${wanted.join(",")}&limit=${MAX_STRUCTURES}`,
  );
  if (!res.ok) throw new Error(`ChEMBL lookup failed (${res.status})`);

  const data = (await res.json()) as { molecules?: ChemblMolecule[] };
  const byId = new Map((data.molecules ?? []).map((m) => [m.molecule_chembl_id, m]));

  const items: StructureItem[] = [];
  const missing: string[] = [];
  // Input order, not response order — the author listed them in an order that
  // probably means something.
  for (const id of wanted) {
    const found = byId.get(id);
    if (!found) {
      missing.push(id);
      continue;
    }
    items.push({
      id,
      name: found.pref_name || id,
      // Absent for biologics: resolved, but nothing to draw.
      smiles: found.molecule_structures?.canonical_smiles ?? "",
    });
  }
  return { items, missing, dropped };
}
