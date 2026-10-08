import { describe, expect, it } from "vitest";

import { diffDocs, inlineDiff, tokenize } from "./pm-diff";
import type { PMDoc } from "./types";

const doc = (...content: unknown[]): PMDoc => ({ type: "doc", content });
const p = (text: string) => ({ type: "paragraph", content: [{ type: "text", text }] });

describe("tokenize", () => {
  it("mirrors ProseMirror position math for nested containers", () => {
    // doc(bulletList(listItem(paragraph("a")))) — PM sizes: p=3, li=5, ul=7; "a" at pos 3
    const d = doc({
      type: "bulletList",
      content: [
        {
          type: "listItem",
          content: [{ type: "paragraph", content: [{ type: "text", text: "a" }] }],
        },
      ],
    });
    const tokens = tokenize(d);
    expect(tokens.map((t) => [t.key, t.from, t.to])).toEqual([
      ["open:bulletList«{}»", 0, 7],
      ["open:listItem«{}»", 1, 6],
      ["open:paragraph«{}»", 2, 5],
      ["a", 3, 4],
    ]);
  });

  it("treats an empty paragraph as a container (size 2), not an atom", () => {
    const tokens = tokenize(doc({ type: "paragraph" }, p("x")));
    // second paragraph opens at pos 2, so the empty one occupied [0,2)
    expect(tokens[1]).toMatchObject({ key: "open:paragraph«{}»", from: 2, to: 5 });
  });
});

describe("diffDocs", () => {
  it("returns empty ranges for identical docs", () => {
    const d = doc(p("hello world"));
    expect(diffDocs(d, d)).toEqual({ left: [], right: [] });
  });

  it("marks an inserted word only on the right, with merged contiguous range", () => {
    const res = diffDocs(doc(p("hello world")), doc(p("hello brave world")));
    expect(res.left).toEqual([]);
    expect(res.right).toEqual([{ from: 7, to: 13, kind: "inline" }]); // "brave "
  });

  it("marks a deleted word only on the left", () => {
    const res = diffDocs(doc(p("hello brave world")), doc(p("hello world")));
    expect(res.left).toEqual([{ from: 7, to: 13, kind: "inline" }]);
    expect(res.right).toEqual([]);
  });

  it("flags a changed embed as a block range on both sides", () => {
    const mol = (smiles: string) => ({ type: "chemStructure", attrs: { smiles } });
    const res = diffDocs(doc(mol("CCO")), doc(mol("CCC")));
    expect(res.left).toEqual([{ from: 0, to: 1, kind: "block" }]);
    expect(res.right).toEqual([{ from: 0, to: 1, kind: "block" }]);
  });

  it("flags an attr-only block change (heading level) without inline noise", () => {
    const h = (level: number) => ({
      type: "heading",
      attrs: { level },
      content: [{ type: "text", text: "Results" }],
    });
    const res = diffDocs(doc(h(2)), doc(h(3)));
    expect(res.left).toEqual([{ from: 0, to: 9, kind: "block" }]);
    expect(res.right).toEqual([{ from: 0, to: 9, kind: "block" }]);
  });

  it("marks an added paragraph as block + inline on the right only", () => {
    const res = diffDocs(doc(p("one")), doc(p("one"), p("two")));
    expect(res.left).toEqual([]);
    expect(res.right).toContainEqual({ from: 5, to: 10, kind: "block" }); // the new paragraph node
    expect(res.right).toContainEqual({ from: 6, to: 9, kind: "inline" }); // "two"
  });

  it("does not bleed a diff across block boundaries", () => {
    // same words, different block membership: the open-sentinels anchor the alignment
    const res = diffDocs(doc(p("alpha"), p("beta")), doc(p("alpha beta")));
    // left: "beta"'s paragraph removed; right: " beta" inserted — but "alpha" untouched
    expect(res.left.every((r) => r.from >= 6)).toBe(true);
    expect(res.right.every((r) => r.from >= 6)).toBe(true);
  });
});

describe("inlineDiff", () => {
  const flat = (segs: { text: string; kind: string }[]) =>
    segs
      .map((s) => (s.kind === "same" ? s.text : `${s.kind === "add" ? "+" : "-"}[${s.text}]`))
      .join("");

  it("interleaves the removed and inserted words in one flow", () => {
    expect(flat(inlineDiff(doc(p("hello cruel world")), doc(p("hello brave world"))))).toBe(
      "hello -[cruel]+[brave] world",
    );
  });

  it("collapses nested block opens to one newline and drops the trailing blank", () => {
    const d = doc(
      p("one"),
      {
        type: "bulletList",
        content: [{ type: "listItem", content: [p("two")] }],
      },
      { type: "paragraph" },
    );
    expect(flat(inlineDiff(d, d))).toBe("one\ntwo");
  });

  it("reports a formatting-only change as no word-level change", () => {
    const h = (level: number) => ({
      type: "heading",
      attrs: { level },
      content: [{ type: "text", text: "Results" }],
    });
    expect(inlineDiff(doc(h(2)), doc(h(3))).every((s) => s.kind === "same")).toBe(true);
  });

  it("marks the whole body as added on a first write", () => {
    expect(flat(inlineDiff({ type: "doc", content: [] }, doc(p("new note"))))).toBe("+[new note]");
  });
});
