import { describe, expect, it } from "vitest";
import type { Verdict } from "../lib/verdict";
import { showsBootstrapExplainer } from "./scorecard-view";

const v = (kind: Verdict["kind"], ci: [number, number] | null): Verdict => ({
  kind,
  headline: "",
  model: 0.7,
  baseline: 0.6,
  delta: 0.1,
  ci,
});

describe("showsBootstrapExplainer", () => {
  it("shows for a comparison with an interval", () => {
    expect(showsBootstrapExplainer(v("beats", [0.6, 0.8]))).toBe(true);
    expect(showsBootstrapExplainer(v("within-noise", [0.6, 0.8]))).toBe(true);
    expect(showsBootstrapExplainer(v("no-better", [0.6, 0.8]))).toBe(true);
  });
  it("hides without an interval or without a comparison", () => {
    expect(showsBootstrapExplainer(v("beats", null))).toBe(false);
    expect(showsBootstrapExplainer(v("is-baseline", [0.6, 0.8]))).toBe(false);
    expect(showsBootstrapExplainer(v("unknown", [0.6, 0.8]))).toBe(false);
  });
});
