import { describe, expect, it } from "vitest";

import { PAGE_SETTINGS_DEFAULTS, pageSettings, pageSettingsClasses } from "./page-settings";

describe("pageSettings", () => {
  it("resolves defaults for null docs and docs saved before settings existed", () => {
    expect(pageSettings(null)).toEqual(PAGE_SETTINGS_DEFAULTS);
    expect(pageSettings(undefined)).toEqual(PAGE_SETTINGS_DEFAULTS);
    expect(pageSettings({ type: "doc" })).toEqual(PAGE_SETTINGS_DEFAULTS);
  });

  it("reads saved values", () => {
    expect(
      pageSettings({ type: "doc", attrs: { layout: "full", textSize: "small", locked: true } }),
    ).toEqual({ layout: "full", textSize: "small", locked: true });
  });

  it("coerces junk values to defaults", () => {
    expect(
      pageSettings({ type: "doc", attrs: { layout: "wide", textSize: 3, locked: "yes" } }),
    ).toEqual(PAGE_SETTINGS_DEFAULTS);
  });

  it("builds wrapper classes per setting", () => {
    expect(pageSettingsClasses(PAGE_SETTINGS_DEFAULTS)).toBe("mx-auto w-full max-w-4xl");
    expect(pageSettingsClasses({ layout: "full", textSize: "small", locked: false })).toBe(
      "w-full tiptap-small",
    );
    expect(pageSettingsClasses({ layout: "standard", textSize: "small", locked: true })).toBe(
      "mx-auto w-full max-w-4xl tiptap-small",
    );
  });
});
