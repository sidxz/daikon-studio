import { describe, expect, it } from "vitest";

import { selectionData } from "./pocket-viewer";

describe("selectionData", () => {
  it("sends each lining residue as a one-long author-numbering range", () => {
    expect(
      selectionData([{ chain: "A", resname: "TYR", resseq: 158 }], { r: 1, g: 2, b: 3 }),
    ).toEqual([
      {
        auth_asym_id: "A",
        start_auth_residue_number: 158,
        end_auth_residue_number: 158,
        color: { r: 1, g: 2, b: 3 },
        sideChain: true,
        representation: "molecular-surface",
        representationColor: { r: 1, g: 2, b: 3 },
      },
    ]);
  });

  it("keeps every residue, including the same number on another chain", () => {
    const data = selectionData([
      { chain: "A", resname: "SER", resseq: 94 },
      { chain: "B", resname: "SER", resseq: 94 },
    ]);
    expect(data.map((d) => [d.auth_asym_id, d.start_auth_residue_number])).toEqual([
      ["A", 94],
      ["B", 94],
    ]);
  });

  it("has nothing to paint for a pocket with no residues", () => {
    expect(selectionData([])).toEqual([]);
  });
});
