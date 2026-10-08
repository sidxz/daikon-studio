import { describe, expect, it, vi } from "vitest";

const fakeMol = {
  is_valid: () => true,
  get_descriptors: () => {
    throw new Error("boom");
  },
  delete: vi.fn(),
};

vi.mock("@/shared/lib/rdkit/rdkit-loader", () => ({
  getRDKit: vi.fn(async () => ({ get_mol: vi.fn(() => fakeMol) })),
}));

import { computeChemSnapshot } from "./molecule-depiction";

describe("computeChemSnapshot", () => {
  it("deletes the RDKit mol even when get_descriptors throws", async () => {
    const result = await computeChemSnapshot("CCO");

    // The outer try/catch intentionally swallows the throw (a snapshot
    // failure must never reject — see the doc comment above the function),
    // so this resolves null rather than rejecting. What this guards against:
    // `mol.delete()` must still run, via the inner `finally`, before that
    // throw propagates past it — i.e. no WASM leak on this path.
    expect(result).toBeNull();
    expect(fakeMol.delete).toHaveBeenCalledTimes(1);
  });
});
