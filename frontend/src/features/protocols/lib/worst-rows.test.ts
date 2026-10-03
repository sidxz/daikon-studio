import { describe, expect, it } from "vitest";
import { orderWorstRows } from "./worst-rows";

const BENZENE = "c1ccccc1";
const PYRIDINE = "c1ccncc1";

const ROWS = [
  { structure: "a", scaffold: BENZENE, residual: 0.95 },
  { structure: "b", scaffold: "C1CC1", residual: -2.1 },
  { structure: "c", scaffold: BENZENE, residual: -1.84 },
  { structure: "d", scaffold: PYRIDINE, residual: 1.41 },
  { structure: "e", scaffold: BENZENE, residual: 1.2 },
  { structure: "f", scaffold: PYRIDINE, residual: -0.88 },
  { structure: "g", scaffold: "", residual: 1.65 },
  { structure: "h", scaffold: "", residual: -1.02 },
];

const order = (mode: "series" | "error") =>
  orderWorstRows(ROWS, mode).map(({ row, series }) => `${row.structure}${series?.label ?? ""}`);

describe("orderWorstRows", () => {
  it("by series: shared scaffolds side by side, biggest series first, then the rest by error", () => {
    expect(order("series")).toEqual(["cA", "eA", "aA", "dB", "fB", "b", "g", "h"]);
  });

  it("by error: worst absolute error first, keeping each compound's series tag", () => {
    expect(order("error")).toEqual(["b", "cA", "g", "dB", "eA", "h", "aA", "fB"]);
  });

  it("never makes a series of compounds with no ring system", () => {
    const tags = orderWorstRows(ROWS, "series").filter(({ row }) => row.scaffold === "");
    expect(tags.every(({ series }) => series === null)).toBe(true);
  });

  it("tags carry the series size, and break a size tie by the worst error", () => {
    const tied = [
      { structure: "p", scaffold: PYRIDINE, residual: 3 },
      { structure: "q", scaffold: PYRIDINE, residual: 0.1 },
      { structure: "r", scaffold: BENZENE, residual: 1 },
      { structure: "s", scaffold: BENZENE, residual: 1 },
    ];
    const [first] = orderWorstRows(tied, "series");
    expect(first.row.structure).toBe("p");
    expect(first.series).toMatchObject({ label: "A", size: 2, scaffold: PYRIDINE, index: 0 });
  });
});
