import { describe, expect, it } from "vitest";

import { parseCssColor } from "./theme";

describe("parseCssColor", () => {
  it("parses the hex forms the design tokens emit", () => {
    expect(parseCssColor("#ffffff")).toEqual({ r: 255, g: 255, b: 255 });
    expect(parseCssColor(" #0f172a ")).toEqual({ r: 15, g: 23, b: 42 }); // dark --ds-surface
    expect(parseCssColor("#FFF")).toEqual({ r: 255, g: 255, b: 255 });
  });

  it("parses rgb()/rgba()", () => {
    expect(parseCssColor("rgb(15, 23, 42)")).toEqual({ r: 15, g: 23, b: 42 });
    expect(parseCssColor("rgba(15, 23, 42, 0.5)")).toEqual({ r: 15, g: 23, b: 42 });
  });

  it("returns null for anything else, so the caller can fall back", () => {
    expect(parseCssColor("")).toBeNull();
    expect(parseCssColor("oklch(0.2 0.04 265)")).toBeNull(); // callers fall back to white
    expect(parseCssColor("#12345")).toBeNull();
  });
});
