import { describe, expect, it } from "vitest";
import { summarySentence } from "./run-chemical-space";

describe("summarySentence", () => {
  it("states how many compounds are inside the domain and the similarity range", () => {
    expect(
      summarySentence({
        total: 3,
        in_domain: 0,
        threshold: 0.3,
        nearest_min: 0.11,
        nearest_max: 0.12,
      }),
    ).toBe(
      "0 of 3 compounds are inside the applicability domain (similarity ≥ 0.3). Nearest training compound: similarity 0.11 to 0.12.",
    );
  });

  it("agrees in number for a single compound and omits a range it does not have", () => {
    expect(
      summarySentence({
        total: 1,
        in_domain: 1,
        threshold: 0.3,
        nearest_min: null,
        nearest_max: null,
      }),
    ).toBe("1 of 1 compound is inside the applicability domain (similarity ≥ 0.3).");
  });

  it("gives one value when every compound is equally close", () => {
    expect(
      summarySentence({
        total: 2,
        in_domain: 2,
        threshold: 0.3,
        nearest_min: 0.5,
        nearest_max: 0.5,
      }),
    ).toBe(
      "2 of 2 compounds are inside the applicability domain (similarity ≥ 0.3). Nearest training compound: similarity 0.50.",
    );
  });
});
