"use client";

import { forwardRef } from "react";

import {
  type SuggestionListRef,
  scrollActiveIntoView,
  useSuggestionKeys,
} from "@/features/pages/lib/editor/suggestion-popup";
import { comboboxOptionClass } from "@/shared/components/ui/combobox";

import type { SlashItem } from "./slash-menu";

export type SlashMenuListRef = SuggestionListRef;

type Props = {
  items: SlashItem[];
  command: (item: SlashItem) => void;
};

/** The `/` popup — same list/keyboard-nav shape as `mention-list.tsx`'s
 *  MentionList, swapping the avatar+email row for a title+hint row. No arrow
 *  glyphs (HARD UI RULE): active state is a background highlight only. */
export const SlashMenuList = forwardRef<SlashMenuListRef, Props>(function SlashMenuList(
  { items, command },
  ref,
) {
  const index = useSuggestionKeys(items, command, ref);

  if (!items.length) {
    return (
      <div className="w-64 rounded-lg border border-border bg-popover px-2 py-1.5 text-sm text-muted-foreground shadow-md">
        No matches
      </div>
    );
  }

  return (
    <div
      role="listbox"
      aria-label="Insert menu"
      className="w-64 overflow-hidden rounded-lg border border-border bg-popover p-1 shadow-md"
    >
      {items.map((it, i) => (
        <button
          key={it.title}
          ref={i === index ? scrollActiveIntoView : null}
          type="button"
          role="option"
          aria-selected={i === index}
          onMouseDown={(e) => {
            e.preventDefault();
            command(it);
          }}
          className={comboboxOptionClass({ active: i === index })}
        >
          <span className="flex min-w-0 flex-col text-left">
            <span className="truncate text-sm text-foreground">{it.title}</span>
            <span className="truncate text-xs text-muted-foreground">{it.hint}</span>
          </span>
        </button>
      ))}
    </div>
  );
});
