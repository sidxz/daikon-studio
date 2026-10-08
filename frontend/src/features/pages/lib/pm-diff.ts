import { diffArrays } from "diff";

import type { PMDoc } from "./types";

export type DiffRange = { from: number; to: number; kind: "inline" | "block" };
export type DocDiff = { left: DiffRange[]; right: DiffRange[] };

/** One pane's worth of a comparison: the ranges to highlight and which side of
 *  the diff they represent (a pane only ever shows its own document's positions
 *  — see `diffDocs`). Lives here rather than beside PageView so both PageView
 *  and PageEditor can name it without importing each other. */
export type DiffProp = { ranges: DiffRange[]; side: "del" | "ins" };

type PMNode = {
  type: string;
  attrs?: Record<string, unknown>;
  content?: PMNode[];
  text?: string;
};

export type Token = { key: string; from: number; to: number; kind: "inline" | "block" };

// Leaf (size-1) node types — a JSON node with no content is otherwise
// indistinguishable from an empty container (size 2). Keep in sync with the
// editor's node set (Task 2 Step 2 verifies the names).
const ATOM_TYPES = new Set([
  "chemStructure",
  "reactionScheme",
  "proteinStructure",
  "sequenceViewer",
  "figureImage",
  "image",
  "entityLink",
  "mention",
  "horizontalRule",
  "hardBreak",
]);

const attrKey = (attrs?: Record<string, unknown>): string =>
  attrs
    ? JSON.stringify(
        Object.fromEntries(Object.entries(attrs).sort(([a], [b]) => a.localeCompare(b))),
      )
    : "{}";

/** Walk a PMDoc, emitting diffable tokens carrying their own ProseMirror
 *  positions. Position math mirrors PM exactly: entering/leaving a container
 *  costs 1 each, a leaf node costs 1, text costs its length. Marks are ignored
 *  (ponytail: mark-only changes don't highlight — add marks to the key if that
 *  ever matters). */
export function tokenize(doc: PMDoc): Token[] {
  const tokens: Token[] = [];
  const walk = (node: PMNode, pos: number): number => {
    if (node.text !== undefined) {
      let offset = 0;
      // word/whitespace runs so jsdiff aligns at word granularity
      for (const part of node.text.split(/(\s+)/)) {
        if (part)
          tokens.push({
            key: part,
            from: pos + offset,
            to: pos + offset + part.length,
            kind: "inline",
          });
        offset += part.length;
      }
      return node.text.length;
    }
    if (ATOM_TYPES.has(node.type)) {
      // identity = type + attrs: an edited SMILES reads as remove + insert
      tokens.push({
        key: `${node.type}«${attrKey(node.attrs)}»`,
        from: pos,
        to: pos + 1,
        kind: "block",
      });
      return 1;
    }
    // container (possibly empty): open-sentinel anchors block alignment and
    // carries the node's full [before, after] span for Decoration.node
    const open: Token = {
      key: `open:${node.type}«${attrKey(node.attrs)}»`,
      from: pos,
      to: pos,
      kind: "block",
    };
    tokens.push(open);
    let size = 2;
    let child = pos + 1;
    for (const c of node.content ?? []) {
      const s = walk(c, child);
      child += s;
      size += s;
    }
    open.to = pos + size;
    return size;
  };
  let pos = 0;
  for (const child of (doc.content ?? []) as PMNode[]) pos += walk(child, pos);
  return tokens;
}

export type DiffSegment = { text: string; kind: "same" | "add" | "del" };

/** The same word-level diff, flattened to one text flow with removals and
 *  insertions interleaved — for surfaces too small for the split panes
 *  `diffDocs` feeds (the datasheet history rows). Structure is lost on purpose:
 *  a block boundary becomes one newline however deeply nested the containers
 *  are, and a leaf node reads as its type. */
export function inlineDiff(oldDoc: PMDoc, newDoc: PMDoc): DiffSegment[] {
  const parts = diffArrays(tokenize(oldDoc), tokenize(newDoc), {
    comparator: (x, y) => x.key === y.key,
  });
  const out: DiffSegment[] = [];
  const push = (text: string, kind: DiffSegment["kind"]) => {
    const prev = out[out.length - 1];
    if (text === "\n" && (!prev || prev.text.endsWith("\n"))) return; // collapse nested opens
    if (prev?.kind === kind) prev.text += text;
    else out.push({ text, kind });
  };
  for (const part of parts) {
    const kind = part.added ? "add" : part.removed ? "del" : "same";
    for (const t of part.value) push(t.kind === "inline" ? t.key : blockText(t.key), kind);
  }
  const last = out[out.length - 1];
  if (last) last.text = last.text.replace(/\n+$/, ""); // the editor's trailing empty paragraph
  return out.filter((s) => s.text !== "");
}

const blockText = (key: string): string =>
  key.startsWith("open:") || key.startsWith("hardBreak")
    ? "\n"
    : `[${key.slice(0, key.indexOf("«"))}]`;

/** Diff two docs. Each side's ranges reference ONLY its own document's
 *  positions — left = removed (old doc), right = inserted (new doc) — which is
 *  what lets two read-only editors highlight without any cross-doc mapping. */
export function diffDocs(oldDoc: PMDoc, newDoc: PMDoc): DocDiff {
  const parts = diffArrays(tokenize(oldDoc), tokenize(newDoc), {
    comparator: (x, y) => x.key === y.key,
  });
  const left: DiffRange[] = [];
  const right: DiffRange[] = [];
  for (const part of parts) {
    if (!part.added && !part.removed) continue;
    const out = part.removed ? left : right;
    for (const t of part.value) {
      const prev = out[out.length - 1];
      // merge contiguous inline runs; block ranges stay one-per-node
      if (t.kind === "inline" && prev?.kind === "inline" && prev.to === t.from) prev.to = t.to;
      else out.push({ from: t.from, to: t.to, kind: t.kind });
    }
  }
  return { left, right };
}
