import Mention from "@tiptap/extension-mention";
import type { Editor, Range } from "@tiptap/react";
import type { SuggestionOptions } from "@tiptap/suggestion";

import { type MentionItem, MentionList } from "./mention-list";
import { suggestionPopup } from "./suggestion-popup";

type ItemsFn = (query: string) => Promise<MentionItem[]>;
type SelectedItem = Pick<MentionItem, "id" | "label" | "kind">;

/** Config for one mention channel (one trigger char). `@` people mentions and
 *  `#` entity mentions are both instances of this — see `makeMention`. */
export type MentionConfig = {
  char: string;
  pluginName: string;
  items: ItemsFn;
  kind?: string;
  /**
   * Overrides the default insert (a `pluginName`-typed mention node) with a
   * custom one — Pages' `#` channel uses this to insert an `entityLink` node
   * instead of a plain mention. Must run its own editor transaction (e.g.
   * `editor.chain().focus().insertContentAt(range, {...}).run()`). Omit for
   * the default mention insert, used by every `@` channel today (card-editor,
   * Pages' people mentions).
   */
  onSelect?: (editor: Editor, range: Range, item: SelectedItem) => void;
};

/**
 * Build a Mention extension for one trigger char. Parameterised so a second
 * instance (char "#", pluginName "entityMention") tags pipeline entities
 * (genes/targets/proteins) instead of people — same suggestion popup, a
 * different insert via `onSelect`.
 */
export function makeMention(opts: MentionConfig) {
  const kind = opts.kind ?? "user";
  return Mention.extend({
    name: opts.pluginName,
    addAttributes() {
      return {
        ...this.parent?.(),
        kind: { default: kind },
      };
    },
  }).configure({
    HTMLAttributes: { class: "mention", "data-mention-kind": opts.pluginName },
    suggestion: buildSuggestion(opts.char, opts.items, opts.onSelect),
  });
}

function buildSuggestion(
  char: string,
  items: ItemsFn,
  onSelect?: MentionConfig["onSelect"],
): Partial<SuggestionOptions> {
  return {
    char,
    items: ({ query }) => items(query),
    // Only override the insert when the caller opts in. An absent `command`
    // key leaves Mention's own rich default (insert a `pluginName`-typed
    // mention node) untouched — Mention spreads this whole object over that
    // default, so `command: undefined` would NOT be equivalent: it'd still be
    // an own key, clobbering the default and silently turning every `@` pick
    // into a no-op. Omitting the key entirely is what keeps `@` working.
    ...(onSelect ? { command: ({ editor, range, props }) => onSelect(editor, range, props) } : {}),
    render: suggestionPopup<MentionItem, SelectedItem>(MentionList),
  };
}
