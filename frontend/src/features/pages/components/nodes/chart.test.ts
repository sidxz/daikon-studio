import { describe, expect, it } from "vitest";

import { Chart } from "./chart";

describe("Chart node", () => {
  it("is an atom block node named chart", () => {
    expect(Chart.name).toBe("chart");
    expect(Chart.config.group).toBe("block");
    expect(Chart.config.atom).toBe(true);
  });

  it("declares every attribute the dialog and renderer read", () => {
    const attrs = Chart.config.addAttributes?.call(Chart as never) ?? {};
    expect(Object.keys(attrs)).toEqual(
      expect.arrayContaining(["kind", "raw", "x", "series", "options", "caption", "width"]),
    );
  });

  it("defaults to a full-width bar chart with no options set", () => {
    const attrs = Chart.config.addAttributes?.call(Chart as never) ?? {};
    const defaults = attrs as Record<string, { default: unknown }>;
    expect(defaults.kind.default).toBe("bar");
    expect(defaults.width.default).toBe("full");
    expect(defaults.options.default).toEqual({});
    // Null, not [] — resolveColumns treats null as "choose for me" and an empty
    // array would mean "the author deselected everything".
    expect(defaults.series.default).toBeNull();
  });
});
