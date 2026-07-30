import { describe, expect, it } from "vitest";
import { buildResultParams } from "./result-query";

describe("AG Grid model -> results API parameters", () => {
  it("sends nothing when nothing is sorted or filtered", () => {
    expect(buildResultParams({ sortModel: [], filterModel: {}, inDomainOnly: false })).toEqual({});
  });

  it("translates a sort", () => {
    expect(
      buildResultParams({
        sortModel: [{ colId: "solubility", sort: "desc" }],
        filterModel: {},
        inDomainOnly: false,
      }),
    ).toEqual({ sort_by: "solubility", sort_dir: "desc" });
  });

  it("takes only the first sort: the API sorts by one column", () => {
    expect(
      buildResultParams({
        sortModel: [
          { colId: "solubility", sort: "asc" },
          { colId: "applicability", sort: "desc" },
        ],
        filterModel: {},
        inDomainOnly: false,
      }),
    ).toEqual({ sort_by: "solubility", sort_dir: "asc" });
  });

  it("translates each number filter operation the columns offer", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          solubility: { filterType: "number", type: "greaterThanOrEqual", filter: 2 },
          uncertainty: { filterType: "number", type: "lessThanOrEqual", filter: 0.5 },
          applicability: { filterType: "number", type: "inRange", filter: 0.2, filterTo: 0.8 },
        },
        inDomainOnly: false,
      }),
    ).toEqual({
      filters: JSON.stringify({
        solubility: { min: 2 },
        uncertainty: { max: 0.5 },
        applicability: { min: 0.2, max: 0.8 },
      }),
    });
  });

  it("turns the in-domain switch into an applicability floor", () => {
    expect(buildResultParams({ sortModel: [], filterModel: {}, inDomainOnly: true })).toEqual({
      filters: JSON.stringify({ applicability: { min: 0.5 } }),
    });
  });

  it("intersects the switch with a user's own applicability filter", () => {
    // Both are active, so both must hold. Taking the tighter bound is the only
    // reading that never shows a compound the switch says to hide.
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          applicability: { filterType: "number", type: "inRange", filter: 0.1, filterTo: 0.7 },
        },
        inDomainOnly: true,
      }),
    ).toEqual({ filters: JSON.stringify({ applicability: { min: 0.5, max: 0.7 } }) });
  });

  it("keeps the user's floor when it is already tighter than the switch", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          applicability: { filterType: "number", type: "greaterThanOrEqual", filter: 0.9 },
        },
        inDomainOnly: true,
      }),
    ).toEqual({ filters: JSON.stringify({ applicability: { min: 0.9 } }) });
  });

  it("drops a filter with no usable bound rather than sending an empty one", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: { solubility: { filterType: "number", type: "greaterThanOrEqual" } },
        inDomainOnly: false,
      }),
    ).toEqual({});
  });
});
