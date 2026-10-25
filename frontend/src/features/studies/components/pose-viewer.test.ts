import { describe, expect, it } from "vitest";
import { combinedPdb, liningResidues } from "./pose-viewer";

const RECEPTOR = [
  "ATOM      1  N   ALA A   1      10.000  11.000  12.000  1.00 20.00           N",
  "ATOM      2  CA  ALA A   1      11.000  11.500  12.500  1.00 20.00           C",
  "REMARK  something the viewer has no use for",
].join("\n");

// PDBQT: PDB's columns up to 54, then a partial charge and an AutoDock atom type.
const POSE = [
  "REMARK VINA RESULT:   -11.5      0.000      0.000",
  "ATOM      1  C   UNL     1      17.229  19.516  10.262  1.00  0.00    +0.020 A ",
  "ATOM      2  N   UNL     1      18.100  20.000  11.000  1.00  0.00    -0.300 N ",
].join("\n");

describe("drawing a pose in its receptor", () => {
  it("keeps the receptor's atoms and drops everything that is not one", () => {
    const out = combinedPdb(RECEPTOR, POSE).split("\n");
    expect(out.filter((l) => l.startsWith("ATOM"))).toHaveLength(2);
    expect(out.some((l) => l.startsWith("REMARK"))).toBe(false);
    expect(out.at(-1)).toBe("END");
  });

  it("rewrites the pose as a heteroatom residue so it is not read as part of the chain", () => {
    const ligand = combinedPdb(RECEPTOR, POSE)
      .split("\n")
      .filter((l) => l.startsWith("HETATM"));
    expect(ligand).toHaveLength(2);
    for (const line of ligand) {
      expect(line).toContain("LIG Z 900");
      // Nothing past the coordinate columns: the partial charge and atom type are not PDB.
      expect(line).not.toContain("+0.020");
      expect(line).not.toContain("-0.300");
    }
  });

  it("carries the coordinates across unchanged, since a moved pose is a wrong answer", () => {
    const first = combinedPdb(RECEPTOR, POSE)
      .split("\n")
      .find((l) => l.startsWith("HETATM"));
    expect(first).toContain("17.229");
    expect(first).toContain("19.516");
    expect(first).toContain("10.262");
  });
});

describe("the residues a pose actually touches", () => {
  const near = "ATOM      1  CA  ASP A  25      17.000  19.516  10.262  1.00 20.00           C";
  const far = "ATOM      2  CA  LEU A  90      60.000  60.000  60.000  1.00 20.00           C";
  const pose = "HETATM    1  C   LIG Z 900      17.229  19.516  10.262  1.00  0.00           C";

  it("takes the lining from where the pose landed, not from the approved pocket", () => {
    const lining = liningResidues([near, far, pose].join("\n"));
    expect(lining).toEqual([{ chain: "A", resseq: 25 }]);
  });

  it("returns nothing when there is no pose to measure against", () => {
    expect(liningResidues([near, far].join("\n"))).toEqual([]);
  });

  it("counts a residue once however many of its atoms are close", () => {
    const second = "ATOM      3  CB  ASP A  25      17.100  19.600  10.300  1.00 20.00           C";
    expect(liningResidues([near, second, pose].join("\n"))).toHaveLength(1);
  });
});
