import { describe, expect, it } from "vitest";

import { ProteinStructure } from "./protein-structure";

describe("ProteinStructure node", () => {
  it("is a block node named proteinStructure with the expected attrs", () => {
    expect(ProteinStructure.name).toBe("proteinStructure");
    const attrs = ProteinStructure.config.addAttributes?.call(ProteinStructure as never) ?? {};
    expect(Object.keys(attrs)).toEqual(
      expect.arrayContaining(["source", "sourceId", "chains", "style", "snapshot", "capturedAt"]),
    );
  });
});
