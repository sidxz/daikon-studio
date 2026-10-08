import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { Extension } from "@tiptap/react";

import type { DiffRange } from "@/features/pages/lib/pm-diff";

/** Static change highlights for a read-only compare pane. The doc never
 *  changes after mount, so the DecorationSet is computed once from the given
 *  ranges — no mapping through transactions. */
export function diffHighlight(ranges: DiffRange[], side: "del" | "ins") {
  return Extension.create({
    name: "diffHighlight",
    addProseMirrorPlugins() {
      return [
        new Plugin({
          key: new PluginKey("diffHighlight"),
          props: {
            decorations(state) {
              const max = state.doc.content.size;
              // ponytail: out-of-bounds ranges are dropped, not thrown — pm-diff's
              // position math is unit-tested, this is the belt to those braces
              return DecorationSet.create(
                state.doc,
                ranges
                  .filter((r) => r.from >= 0 && r.to <= max && r.from < r.to)
                  .map((r) =>
                    r.kind === "block"
                      ? Decoration.node(r.from, r.to, { class: `diff-block-${side}` })
                      : Decoration.inline(r.from, r.to, { class: `diff-${side}` }),
                  ),
              );
            },
          },
        }),
      ];
    },
  });
}
