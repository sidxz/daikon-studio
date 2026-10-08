import { SplitStrategy } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import { SPLIT_VOCABULARY, isGroupedSplit, splitVocabulary } from "./split";

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
});
