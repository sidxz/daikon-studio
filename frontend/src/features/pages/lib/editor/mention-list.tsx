"use client";

import { forwardRef } from "react";

import { initials } from "@/shared/components/initials-avatar";
import { Avatar, AvatarFallback } from "@/shared/components/ui/avatar";
import { comboboxOptionClass } from "@/shared/components/ui/combobox";

import {
  type SuggestionListRef,
  scrollActiveIntoView,
  useSuggestionKeys,
} from "./suggestion-popup";

/** One selectable suggestion in a mention popup — a person (`@`) today, an
 *  entity like a gene/target (`#`) later. `email`/`avatarUrl` are person-only;
 *  `kind` distinguishes mention channels sharing one list UI. */
export type MentionItem = {
  id: string;
  label: string;
  email?: string;
  avatarUrl?: string | null;
  kind?: string;
};

export type MentionListRef = SuggestionListRef;

type Props = {
  items: MentionItem[];
  command: (item: Pick<MentionItem, "id" | "label" | "kind">) => void;
};

export const MentionList = forwardRef<MentionListRef, Props>(function MentionList(
  { items, command },
  ref,
) {
  // `kind` rides along so a channel with a custom insert (Pages' `#`) knows
  // which entity type was picked — harmless for `@` people (kind is always
  // undefined there, same as before this field was added).
  const select = (it: MentionItem) => command({ id: it.id, label: it.label, kind: it.kind });
  const index = useSuggestionKeys(items, select, ref);

  if (!items.length) {
    return (
      <div className="w-56 rounded-lg border border-border bg-popover px-2 py-1.5 text-sm text-muted-foreground shadow-md">
        No matches
      </div>
    );
  }

  return (
    <div
      role="listbox"
      aria-label="Mention suggestions"
      className="w-56 overflow-hidden rounded-lg border border-border bg-popover p-1 shadow-md"
    >
      {items.map((it, i) => (
        <button
          key={it.id}
          ref={i === index ? scrollActiveIntoView : null}
          type="button"
          role="option"
          aria-selected={i === index}
          onMouseDown={(e) => {
            e.preventDefault();
            select(it);
          }}
          className={comboboxOptionClass({ active: i === index })}
        >
          <Avatar className="size-6">
            <AvatarFallback className="text-[10px] font-medium">
              {initials(it.label)}
            </AvatarFallback>
          </Avatar>
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-sm text-foreground">{it.label}</span>
            <span className="truncate text-xs text-muted-foreground">{it.email}</span>
          </span>
        </button>
      ))}
    </div>
  );
});
