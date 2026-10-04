import { describe, expect, it } from "vitest";
import { formatCutoff } from "./format-cutoff";

describe("formatCutoff", () => {
  it("shows three significant digits without trailing zeros", () => {
    expect(formatCutoff(0.031)).toBe("0.031");
    expect(formatCutoff(0.5)).toBe("0.5");
    expect(formatCutoff(0.9867)).toBe("0.987");
    expect(formatCutoff(0.12345)).toBe("0.123");
  });

  it("never shows a value below 1 as 1", () => {
    expect(formatCutoff(0.99997)).toBe("0.99997");
    expect(formatCutoff(0.9996)).toBe("0.9996");
    expect(formatCutoff(0.99999994)).toBe("0.9999999");
  });

  it("shows a cutoff of exactly 1 as 1", () => {
    expect(formatCutoff(1)).toBe("1");
  });
});
