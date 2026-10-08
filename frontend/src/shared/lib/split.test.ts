import { SplitStrategy } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import {
  SPLIT_VOCABULARY,
  hasRandomComparison,
  isGroupedSplit,
  scaffoldSeparationNote,
  splitFacts,
  splitVocabulary,
} from "./split";

describe("split vocabulary", () => {
  it("covers every strategy the API can send", () => {
    for (const strategy of Object.values(SplitStrategy)) {
      expect(SPLIT_VOCABULARY[strategy]).toBeDefined();
    }
  });

  it("calls every split but random a grouped one", () => {
    expect(isGroupedSplit("random")).toBe(false);
    expect(isGroupedSplit("scaffold")).toBe(true);
    expect(isGroupedSplit("identity")).toBe(true);
    expect(isGroupedSplit("position")).toBe(true);
  });

  it("says what each split held out, and only that", () => {
    expect(SPLIT_VOCABULARY.scaffold.held).toContain("scaffold");
    expect(SPLIT_VOCABULARY.identity.held).toContain("protein family");
    expect(SPLIT_VOCABULARY.position.held).toContain("residue position");
    // The bug this module exists for: a sequence split printing a claim about
    // scaffolds, which is not what it separated.
    expect(SPLIT_VOCABULARY.identity.held).not.toContain("scaffold");
    expect(SPLIT_VOCABULARY.position.held).not.toContain("scaffold");
  });

  it("claims nothing specific about a strategy it has never heard of", () => {
    // `split_strategy` is a bare string on the wire, so this is reachable.
    const unknown = splitVocabulary("butina");
    expect(unknown.held).not.toContain("scaffold");
    expect(isGroupedSplit("butina")).toBe(true);
  });

  it("has a reason-for-no-gap for every strategy, and never claims a random split wrongly", () => {
    for (const strategy of Object.values(SplitStrategy)) {
      expect(SPLIT_VOCABULARY[strategy].noGap).toBeTruthy();
    }
    expect(SPLIT_VOCABULARY.predefined.noGap).not.toContain("scored on a random split");
  });
});

describe("scaffoldSeparationNote", () => {
  const shared = "12 compounds have a scaffold present in both training and test sets.";

  it("tells a scaffold split that sharing a scaffold is a failure", () => {
    expect(scaffoldSeparationNote("scaffold", shared, true)).toContain("should prevent this");
    expect(scaffoldSeparationNote("scaffold", shared, false)).toContain("As expected");
  });

  it("tells a random split that sharing a scaffold is expected", () => {
    expect(scaffoldSeparationNote("random", shared, true)).toContain("Expected for a random split");
  });

  it("names the group for a split that groups something", () => {
    expect(scaffoldSeparationNote("identity", shared, true)).toContain("protein family");
  });

  it("never says a split groups by null", () => {
    // `predefined` groups nothing, and the old code interpolated `group` straight into
    // the sentence -- which compiles, and prints "groups by null" to a scientist.
    const note = scaffoldSeparationNote("predefined", shared, true);
    expect(note).not.toContain("null");
    expect(note).toContain("from your file");
  });
});

describe("splitFacts", () => {
  it("describes a computed split by its seed and intended fractions", () => {
    const facts = splitFacts({ strategy: "scaffold", seed: 42, fractions: [0.8, 0.1, 0.1] });
    expect(facts.map((fact) => fact.label)).toEqual(["Seed", "Train / validation / test"]);
    expect(facts[1].value).toBe("0.8 / 0.1 / 0.1");
  });

  it("describes a predefined split by its column, and never by fractions it ignored", () => {
    // The bug this exists for: the detail page printed "0.8 / 0.1 / 0.1" for a split
    // that was actually 2,651 / 0 / 666. Those fractions are inert for a predefined
    // split, and stating them tells the reader a validation partition exists.
    const facts = splitFacts({
      strategy: "predefined",
      seed: 42,
      fractions: [0.8, 0.1, 0.1],
      column: "split",
    });
    expect(facts).toEqual([{ label: "Partition column", value: "split" }]);
  });

  it("falls back to the default fractions when the API omits them", () => {
    const facts = splitFacts({ strategy: "random", seed: 7 });
    expect(facts[1].value).toBe("0.8 / 0.1 / 0.1");
  });
});

describe("hasRandomComparison", () => {
  it("matches the backend rule, which is 'not random' rather than 'grouped'", () => {
    // The backend runs the comparison leg for every strategy but RANDOM. `predefined`
    // groups nothing, so `isGroupedSplit` is false for it -- and the form used to
    // promise two stages while three ran.
    expect(hasRandomComparison("random")).toBe(false);
    expect(hasRandomComparison("scaffold")).toBe(true);
    expect(hasRandomComparison("identity")).toBe(true);
    expect(hasRandomComparison("position")).toBe(true);
    expect(hasRandomComparison("predefined")).toBe(true);
  });

  it("assumes an unknown strategy does get one, matching the backend default", () => {
    expect(hasRandomComparison("butina")).toBe(true);
  });
});
