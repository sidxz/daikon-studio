import { describe, expect, it } from "vitest";

import {
  PIE_MAX_SLICES,
  SERIES_COLORS,
  SERIES_SHAPES,
  axisRange,
  axisTitles,
  dataExtent,
  dateTickFormat,
  foldPieSlices,
  groupRowsBy,
  segmentFor,
  toScatterPoints,
} from "./chart-figure";

describe("SERIES_COLORS", () => {
  it("is the five design-token slots in fixed order", () => {
    expect(SERIES_COLORS).toEqual([
      "var(--chart-1)",
      "var(--chart-2)",
      "var(--chart-3)",
      "var(--chart-4)",
      "var(--chart-5)",
    ]);
  });
});

describe("foldPieSlices", () => {
  const rows = Array.from({ length: 9 }, (_, i) => ({ name: `s${i}`, v: i + 1 }));

  it("leaves a short list untouched", () => {
    const short = rows.slice(0, 3);
    expect(foldPieSlices(short, "name", "v")).toEqual(short);
  });

  it("keeps the largest slices and folds the rest into one Other", () => {
    const folded = foldPieSlices(rows, "name", "v");
    expect(folded).toHaveLength(PIE_MAX_SLICES);
    // 9,8,7,6,5 kept; 4+3+2+1 = 10 folded.
    expect(folded.slice(0, PIE_MAX_SLICES - 1).map((r) => r.name)).toEqual([
      "s8",
      "s7",
      "s6",
      "s5",
      "s4",
    ]);
    expect(folded[PIE_MAX_SLICES - 1]).toEqual({ name: "Other", v: 10 });
  });

  it("ignores non-numeric values when folding", () => {
    const messy = [...rows, { name: "bad", v: "n/a" }];
    const folded = foldPieSlices(messy, "name", "v");
    expect(folded[PIE_MAX_SLICES - 1].v).toBe(10);
  });
});

describe("toScatterPoints", () => {
  const rows = [
    { conc: 1, response: 4, sd: 1 },
    { conc: 10, response: 22, sd: 3 },
  ];

  // Regression: the series column must become the point's `y`, never the
  // Scatter's dataKey — Recharts reads that as the point-size channel and
  // renders every marker as a ~2px speck.
  it("remaps each row onto the {x, y} shape the axes read", () => {
    expect(toScatterPoints(rows, "conc", "response")).toEqual([
      { x: 1, y: 4, e: undefined },
      { x: 10, y: 22, e: undefined },
    ]);
  });

  it("carries the error column through as `e` when one is chosen", () => {
    expect(toScatterPoints(rows, "conc", "response", "sd").map((p) => p.e)).toEqual([1, 3]);
  });
});

describe("dataExtent", () => {
  it("spans every plotted value, x and series alike", () => {
    const rows = [
      { observed: 5.1, predicted: 5.4 },
      { observed: 7.9, predicted: 8.1 },
    ];
    expect(dataExtent(rows, "observed", ["predicted"])).toEqual([5.1, 8.1]);
  });

  it("ignores non-numeric cells and falls back to the unit range when empty", () => {
    expect(dataExtent([{ a: "x", b: 3 }], "a", ["b"])).toEqual([3, 3]);
    expect(dataExtent([], "a", ["b"])).toEqual([0, 1]);
  });
});

describe("segmentFor", () => {
  // Regression: "dataMin"/"dataMax" are resolved for an axis `domain` but NOT
  // inside a ReferenceLine `segment`, so y=x has to be real coordinates.
  it("draws identity across the data's own extent", () => {
    expect(segmentFor("identity", [5.1, 8.1])).toEqual([
      { x: 5.1, y: 5.1 },
      { x: 8.1, y: 8.1 },
    ]);
  });

  it("keeps the chance diagonal on the unit square regardless of the data", () => {
    expect(segmentFor("diagonal", [5.1, 8.1])).toEqual([
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ]);
  });
});

describe("groupRowsBy", () => {
  const rows = [
    { Temperature: 150.5, Yield: 45.2, Catalyst: "Catalyst_A" },
    { Temperature: 160.0, Yield: 52.1, Catalyst: "Catalyst_A" },
    { Temperature: 155.2, Yield: 38.9, Catalyst: "Catalyst_B" },
    { Temperature: 162.8, Yield: 41.5, Catalyst: "Catalyst_B" },
  ];

  it("splits rows into one group per distinct value", () => {
    const groups = groupRowsBy(rows, "Catalyst");
    expect(groups.map((g) => g.key)).toEqual(["Catalyst_A", "Catalyst_B"]);
    expect(groups[0].rows).toHaveLength(2);
    expect(groups[1].rows.map((r) => r.Yield)).toEqual([38.9, 41.5]);
  });

  it("orders groups by first appearance, not alphabetically", () => {
    const reordered = [rows[2], rows[0]];
    expect(groupRowsBy(reordered, "Catalyst").map((g) => g.key)).toEqual([
      "Catalyst_B",
      "Catalyst_A",
    ]);
  });

  it("caps at the number of hue/shape slots", () => {
    const many = Array.from({ length: 9 }, (_, i) => ({ g: `g${i}`, v: i }));
    expect(groupRowsBy(many, "g")).toHaveLength(SERIES_COLORS.length);
  });

  it("buckets missing values under one empty key rather than dropping them", () => {
    expect(groupRowsBy([{ v: 1 }, { v: 2 }], "absent").map((g) => g.key)).toEqual([""]);
  });
});

describe("SERIES_SHAPES", () => {
  // Shape is a redundant encoding of the colour grouping, so the two lists must
  // stay the same length or a group would get a colour with no shape.
  it("pairs one-to-one with SERIES_COLORS", () => {
    expect(SERIES_SHAPES).toHaveLength(SERIES_COLORS.length);
    expect(new Set(SERIES_SHAPES).size).toBe(SERIES_SHAPES.length);
  });
});

describe("axisRange", () => {
  const FIT = ["dataMin", "dataMax"] as const;

  it("leaves the axis fitted when neither bound is given", () => {
    expect(axisRange(null, null, FIT)).toEqual({ domain: FIT });
    expect(axisRange(undefined, undefined, FIT).allowDataOverflow).toBeUndefined();
  });

  // allowDataOverflow is what makes a pin binding: without it Recharts widens
  // the domain to fit stray points, so "lock x to 0-100" silently becomes 0-120.
  it("pins both ends and clips out-of-range points", () => {
    expect(axisRange(0, 100, FIT)).toEqual({
      domain: [0, 100],
      allowDataOverflow: true,
    });
  });

  it("pins one end and leaves the other automatic", () => {
    expect(axisRange(0, null, FIT).domain).toEqual([0, "dataMax"]);
    expect(axisRange(null, 100, FIT).domain).toEqual(["dataMin", 100]);
  });

  it("treats zero as a real bound, not as absent", () => {
    // The guard is `!= null`, so a falsy 0 must still pin the axis.
    expect(axisRange(0, 0, FIT).allowDataOverflow).toBe(true);
    expect(axisRange(0, null, FIT).domain[0]).toBe(0);
  });

  it("honours a different fallback for zero-baseline charts", () => {
    expect(axisRange(null, null, [0, "auto"]).domain).toEqual([0, "auto"]);
    expect(axisRange(null, 50, [0, "auto"]).domain).toEqual([0, 50]);
  });
});

describe("axisTitles", () => {
  it("defaults to the column names already in the data", () => {
    expect(axisTitles({}, "conc_nm", ["inhibition"])).toEqual({
      x: "conc_nm",
      y: "inhibition",
    });
  });

  it("leaves the y title blank for several series, because the legend names them", () => {
    expect(axisTitles({}, "epoch", ["train", "validation"]).y).toBe("");
  });

  it("lets an author override either title", () => {
    expect(axisTitles({ xLabel: "Concentration (nM)" }, "conc_nm", ["v"]).x).toBe(
      "Concentration (nM)",
    );
  });

  it("treats an explicit empty string as deliberately cleared, not as absent", () => {
    // `??` not `||`: "" must survive rather than falling back to the column name.
    expect(axisTitles({ xLabel: "", yLabel: "" }, "conc_nm", ["v"])).toEqual({ x: "", y: "" });
  });
});

describe("dateTickFormat", () => {
  const day = 86_400_000;

  it("coarsens as the span grows", () => {
    expect(dateTickFormat(4 * day)).toBe("d MMM");
    expect(dateTickFormat(200 * day)).toBe("MMM yyyy");
    expect(dateTickFormat(5 * 365 * day)).toBe("yyyy");
  });

  it("falls back to clock time inside a couple of days", () => {
    expect(dateTickFormat(6 * 3_600_000)).toBe("HH:mm");
  });
});
