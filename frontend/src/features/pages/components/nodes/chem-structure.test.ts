import { describe, expect, it } from "vitest";

import { ChemStructure } from "./chem-structure";

describe("ChemStructure node", () => {
  it("is a block node named chemStructure with the expected attrs", () => {
    expect(ChemStructure.name).toBe("chemStructure");
    const attrs = ChemStructure.config.addAttributes?.call(ChemStructure as never) ?? {};
    expect(Object.keys(attrs)).toEqual(
      expect.arrayContaining(["source", "smiles", "sourceId", "snapshot", "capturedAt"]),
    );
  });
});
