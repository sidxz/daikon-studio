import { describe, expect, it } from "vitest";

import { type EmbedKind, buildSlashItems, filterSlashItems } from "./slash-menu";

describe("slash menu registry", () => {
  it("offers every generic block and embed, and nothing else", () => {
    const items = buildSlashItems(() => {});
    const titles = items.map((i) => i.title.toLowerCase());
    for (const expected of [
      "table",
      "code",
      "task list",
      "figure",
      "molecule",
      "reaction",
      "protein",
      "sequence",
      "equation",
    ]) {
      expect(titles.some((t) => t.includes(expected))).toBe(true);
    }
    // There is no callout node in the schema — the menu must not offer one.
    expect(titles.some((t) => t.includes("callout"))).toBe(false);
  });

  it("narrows the list by query (title or keyword match)", () => {
    const items = buildSlashItems(() => {});
    expect(filterSlashItems(items, "").length).toBe(items.length);
    expect(filterSlashItems(items, "prot").map((i) => i.title)).toEqual(["Protein"]);
    // Case-insensitive, and matches a keyword as well as the title.
    expect(filterSlashItems(items, "SMILES").map((i) => i.title)).toEqual(["Molecule"]);
    expect(filterSlashItems(items, "does-not-exist")).toEqual([]);
  });

  it("routes each embed item to the matching dialog kind via the callback, not the editor", () => {
    const opened: EmbedKind[] = [];
    const items = buildSlashItems((kind) => opened.push(kind));
    const fakeEditor = {} as never; // embeds never touch the editor directly
    for (const kind of ["figure", "molecule", "reaction", "protein", "sequence"] as const) {
      const item = items.find((i) => i.title.toLowerCase() === kind);
      item?.run(fakeEditor);
    }
    expect(opened).toEqual(["figure", "molecule", "reaction", "protein", "sequence"]);
  });

  it("runs generic-block commands directly against the editor's chain", () => {
    // A minimal chainable fake: every method (including `focus`/`run`) records its
    // name and returns the same proxy so `.chain().focus().insertTable(...).run()`
    // resolves without a real TipTap editor.
    const calls: string[] = [];
    const chain: Record<string, (...args: unknown[]) => unknown> = new Proxy(
      {},
      {
        get(_t, prop: string) {
          calls.push(prop);
          return () => chain;
        },
      },
    );
    const fakeEditor = { chain: () => chain } as never;

    const items = buildSlashItems(() => {});
    items.find((i) => i.title === "Table")?.run(fakeEditor);
    expect(calls).toEqual(["focus", "insertTable", "run"]);

    calls.length = 0;
    items.find((i) => i.title === "Code block")?.run(fakeEditor);
    expect(calls).toEqual(["focus", "toggleCodeBlock", "run"]);

    calls.length = 0;
    items.find((i) => i.title === "Task list")?.run(fakeEditor);
    expect(calls).toEqual(["focus", "toggleTaskList", "run"]);
  });
});

describe("chart items", () => {
  it("offers a plain Chart plus one item per preset", () => {
    const titles = buildSlashItems(() => {}).map((i) => i.title);
    expect(titles).toContain("Chart");
    for (const label of [
      "Volcano plot",
      "ROC curve",
      "Dose-response",
      "Parity plot",
      "Feature importance",
      "Kaplan-Meier",
      "Property profile",
      "Composition",
    ]) {
      expect(titles).toContain(label);
    }
  });

  it("opens the chart dialog with the matching preset id", () => {
    const calls: [EmbedKind, unknown, string | undefined][] = [];
    const items = buildSlashItems((kind, editing, preset) => calls.push([kind, editing, preset]));
    filterSlashItems(items, "volcano")[0].run({} as never);
    expect(calls).toEqual([["chart", undefined, "volcano"]]);
  });

  it("finds a preset by a domain keyword, not just its label", () => {
    const items = buildSlashItems(() => {});
    expect(filterSlashItems(items, "ic50").map((i) => i.title)).toEqual(["Dose-response"]);
    expect(filterSlashItems(items, "auc").map((i) => i.title)).toEqual(["ROC curve"]);
  });
});

describe("structure grid item", () => {
  it("opens the structureGrid dialog rather than inserting a bare node", () => {
    const calls: EmbedKind[] = [];
    const items = buildSlashItems((kind) => calls.push(kind));
    items.find((i) => i.title === "Structure grid")?.run({} as never);
    expect(calls).toEqual(["structureGrid"]);
  });

  it("is findable by the words a chemist would type, not just its title", () => {
    const items = buildSlashItems(() => {});
    // A mark with no keywords is unreachable by any word but its own title —
    // which is why "/scatter" finds no chart today. Don't repeat that here.
    for (const word of ["structures", "compounds", "chemcellar", "chembl", "sar"]) {
      expect(filterSlashItems(items, word).map((i) => i.title)).toContain("Structure grid");
    }
  });

  it("leaves the single-molecule item as the sole SMILES match", () => {
    // "smiles" is deliberately NOT a keyword here: a one-structure insert is
    // what someone typing it wants, and the grid already answers "structures".
    const items = buildSlashItems(() => {});
    expect(filterSlashItems(items, "SMILES").map((i) => i.title)).toEqual(["Molecule"]);
  });
});

describe("equation item", () => {
  it("opens the math dialog, found by 'latex' keyword", () => {
    const calls: EmbedKind[] = [];
    const items = buildSlashItems((kind) => calls.push(kind));
    filterSlashItems(items, "latex")[0].run({} as never);
    expect(calls).toEqual(["math"]);
  });
});
