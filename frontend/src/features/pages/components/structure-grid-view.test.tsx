import { describe, expect, it } from "vitest";

import { SOURCE_LABEL, provenanceLine } from "./structure-grid-view";

describe("SOURCE_LABEL", () => {
  it("names all four sources", () => {
    expect(Object.keys(SOURCE_LABEL).sort()).toEqual([
      "chembl",
      "chemcellar",
      "collection",
      "smiles",
    ]);
  });
});

describe("provenanceLine", () => {
  it("reads as a sentence with the source, the count and the capture date", () => {
    expect(provenanceLine("chemcellar", 3, "2026-07-23T10:00:00.000Z")).toBe(
      "3 structures from ChemCellar · captured 23 Jul 2026",
    );
  });

  it("uses the singular for one structure", () => {
    expect(provenanceLine("smiles", 1, "2026-07-23T10:00:00.000Z")).toBe(
      "1 structure from SMILES · captured 23 Jul 2026",
    );
  });

  it("omits the capture clause when the node predates capturedAt", () => {
    expect(provenanceLine("smiles", 2, null)).toBe("2 structures from SMILES");
  });

  it("omits the capture clause rather than printing Invalid Date", () => {
    expect(provenanceLine("smiles", 2, "not-a-date")).toBe("2 structures from SMILES");
  });

  it("falls back to the raw source for an unknown value", () => {
    expect(provenanceLine("future-source", 1, null)).toBe("1 structure from future-source");
  });
});
