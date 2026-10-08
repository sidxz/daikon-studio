import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MAX_STRUCTURES,
  capStructures,
  fetchChemblStructures,
  parseStructureList,
} from "./molecules";

describe("parseStructureList", () => {
  it("reads two positional columns as name then SMILES", () => {
    expect(parseStructureList("Erlotinib,CCO\nGefitinib,CCN")).toEqual([
      { name: "Erlotinib", smiles: "CCO" },
      { name: "Gefitinib", smiles: "CCN" },
    ]);
  });

  it("prefers tabs when any line contains one (Excel paste)", () => {
    // The name legitimately contains a comma, so comma-splitting would be wrong.
    expect(parseStructureList("Compound A, batch 2\tCCO")).toEqual([
      { name: "Compound A, batch 2", smiles: "CCO" },
    ]);
  });

  it("uses a header row to locate columns, in either order", () => {
    expect(parseStructureList("smiles,name\nCCO,Ethanol")).toEqual([
      { name: "Ethanol", smiles: "CCO" },
    ]);
    expect(parseStructureList("name,smiles\nEthanol,CCO")).toEqual([
      { name: "Ethanol", smiles: "CCO" },
    ]);
  });

  it("treats 'structure' as a header keyword too", () => {
    expect(parseStructureList("name,structure\nEthanol,CCO")).toEqual([
      { name: "Ethanol", smiles: "CCO" },
    ]);
  });

  it("does not mistake a data row for a header", () => {
    // No cell is named smiles/structure, so all three lines are compounds.
    expect(parseStructureList("A,CCO\nB,CCN\nC,CCC")).toHaveLength(3);
  });

  it("skips blank lines and rows with no SMILES", () => {
    expect(parseStructureList("A,CCO\n\nB,\n  \nC,CCC")).toEqual([
      { name: "A", smiles: "CCO" },
      { name: "C", smiles: "CCC" },
    ]);
  });

  it("falls back to the SMILES as the name when only one column is given", () => {
    expect(parseStructureList("CCO\nCCN")).toEqual([
      { name: "CCO", smiles: "CCO" },
      { name: "CCN", smiles: "CCN" },
    ]);
  });
});

describe("capStructures", () => {
  it("passes a short list through with nothing dropped", () => {
    expect(capStructures([1, 2, 3])).toEqual({ items: [1, 2, 3], dropped: 0 });
  });

  it("trims to the cap and reports how many were dropped", () => {
    const many = Array.from({ length: MAX_STRUCTURES + 7 }, (_, i) => i);
    const capped = capStructures(many);
    expect(capped.items).toHaveLength(MAX_STRUCTURES);
    expect(capped.dropped).toBe(7);
  });
});

describe("fetchChemblStructures", () => {
  afterEach(() => vi.unstubAllGlobals());

  /** Records the URLs it was called with. Reading them from a closure rather
   *  than `mock.calls[0][0]`: a zero-arg `vi.fn` types its calls as an empty
   *  tuple, so indexing one is a type error. */
  function stubChembl(molecules: unknown[]) {
    const urls: string[] = [];
    const fetchMock = vi.fn(async (url: string) => {
      urls.push(url);
      return { ok: true, json: async () => ({ molecules }) };
    });
    vi.stubGlobal("fetch", fetchMock);
    return { fetchMock, urls };
  }

  it("resolves a whole list in ONE request, preserving input order", async () => {
    const { fetchMock, urls } = stubChembl([
      {
        molecule_chembl_id: "CHEMBL192",
        pref_name: "SILDENAFIL",
        molecule_structures: { canonical_smiles: "CCC" },
      },
      {
        molecule_chembl_id: "CHEMBL25",
        pref_name: "ASPIRIN",
        molecule_structures: { canonical_smiles: "CCO" },
      },
    ]);

    const out = await fetchChemblStructures(["chembl25", " CHEMBL192 "]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(urls[0]).toContain("molecule_chembl_id__in=CHEMBL25,CHEMBL192");
    // Input order, not response order.
    expect(out.items).toEqual([
      { id: "CHEMBL25", name: "ASPIRIN", smiles: "CCO" },
      { id: "CHEMBL192", name: "SILDENAFIL", smiles: "CCC" },
    ]);
    expect(out.missing).toEqual([]);
  });

  it("keeps a record that resolved without a structure (a biologic)", async () => {
    // CHEMBL1201585 (trastuzumab) is a real record with no molecule_structures.
    stubChembl([{ molecule_chembl_id: "CHEMBL1201585", pref_name: "TRASTUZUMAB" }]);

    const out = await fetchChemblStructures(["CHEMBL1201585"]);

    // Resolved, not missing — the card shows the name over a "no structure" slot.
    expect(out.items).toEqual([{ id: "CHEMBL1201585", name: "TRASTUZUMAB", smiles: "" }]);
    expect(out.missing).toEqual([]);
  });

  it("reports unresolved ids by id rather than dropping them silently", async () => {
    stubChembl([
      {
        molecule_chembl_id: "CHEMBL25",
        pref_name: "ASPIRIN",
        molecule_structures: { canonical_smiles: "CCO" },
      },
    ]);

    const out = await fetchChemblStructures(["CHEMBL25", "CHEMBL999999999"]);

    expect(out.items).toHaveLength(1);
    expect(out.missing).toEqual(["CHEMBL999999999"]);
  });

  it("falls back to the id when ChEMBL has no preferred name", async () => {
    stubChembl([
      {
        molecule_chembl_id: "CHEMBL25",
        pref_name: null,
        molecule_structures: { canonical_smiles: "CCO" },
      },
    ]);
    const out = await fetchChemblStructures(["CHEMBL25"]);
    expect(out.items[0]?.name).toBe("CHEMBL25");
  });

  it("de-duplicates and caps before building the URL", async () => {
    const { urls } = stubChembl([]);
    const ids = Array.from({ length: MAX_STRUCTURES + 10 }, (_, i) => `CHEMBL${i}`);

    const out = await fetchChemblStructures([...ids, "CHEMBL0"]);

    expect(out.dropped).toBe(10);
    const url = String(urls[0]);
    expect(url).not.toContain(`CHEMBL${MAX_STRUCTURES}`);
    expect(url).toContain(`limit=${MAX_STRUCTURES}`);
  });

  it("makes no request at all for an empty list", async () => {
    const { fetchMock } = stubChembl([]);
    expect(await fetchChemblStructures([" ", ""])).toEqual({
      items: [],
      missing: [],
      dropped: 0,
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("throws on a non-ok response so the dialog can show an error state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) })),
    );
    await expect(fetchChemblStructures(["CHEMBL25"])).rejects.toThrow(/503/);
  });
});
