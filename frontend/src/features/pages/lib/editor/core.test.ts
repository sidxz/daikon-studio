import { describe, expect, it, vi } from "vitest";

import { sharedExtensions } from "./core";

describe("sharedExtensions", () => {
  it("includes StarterKit, Placeholder, TaskList, TaskItem", () => {
    const names = sharedExtensions({ placeholder: "x" }).map((e) => e.name);
    expect(names).toContain("starterKit");
    expect(names).toContain("placeholder");
    expect(names).toContain("taskList");
    expect(names).toContain("taskItem");
  });

  it("adds one mention extension per config with the given pluginName", () => {
    const exts = sharedExtensions({
      placeholder: "x",
      mentions: [{ char: "@", pluginName: "mention", items: async () => [] }],
    });
    expect(exts.map((e) => e.name)).toContain("mention");
  });

  // Backward-compat guard for makeMention's `onSelect` seam (Task 12): Mention
  // spreads the configured suggestion over its own rich default `command`, so
  // an explicit `command: undefined` would clobber that default and silently
  // turn every `@` pick into a no-op. Proving the key is genuinely absent (not
  // just falsy) is what keeps `@` — used by card-editor and Pages alike —
  // working exactly as before.
  it("without onSelect, the mention's suggestion has no command override", () => {
    const [ext] = sharedExtensions({
      placeholder: "x",
      mentions: [{ char: "@", pluginName: "mention", items: async () => [] }],
    }).filter((e) => e.name === "mention");
    expect(Object.hasOwn(ext.options.suggestion, "command")).toBe(false);
  });

  it("with onSelect, the suggestion command delegates to it instead of the default insert", () => {
    const onSelect = vi.fn();
    const [ext] = sharedExtensions({
      placeholder: "x",
      mentions: [{ char: "#", pluginName: "entityMention", items: async () => [], onSelect }],
    }).filter((e) => e.name === "entityMention");
    const fakeEditor = {} as never;
    const fakeRange = { from: 0, to: 1 };
    const item = { id: "P1", label: "KRAS", kind: "protein" };

    ext.options.suggestion.command?.({ editor: fakeEditor, range: fakeRange, props: item });

    expect(onSelect).toHaveBeenCalledWith(fakeEditor, fakeRange, item);
  });
});
