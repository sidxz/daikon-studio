import Placeholder from "@tiptap/extension-placeholder";
import TaskItem from "@tiptap/extension-task-item";
import TaskList from "@tiptap/extension-task-list";
import type { AnyExtension } from "@tiptap/react";
import StarterKit, { type StarterKitOptions } from "@tiptap/starter-kit";

import { type MentionConfig, makeMention } from "./mention";

/** Shared prose styling + SSR convention. Every editor built on this core sets
 *  `immediatelyRender: false` in its useEditor call (Next SSR hydration). */
export const EDITOR_PROSE_CLASS =
  "tiptap min-h-96 rounded-b-lg border border-border px-3 py-2 focus:outline-none";

// `useEditor({ extensions })` takes `AnyExtension[]` (tiptap's Extension | Node |
// Mark union) — not the narrower `Extension` class, which only StarterKit and
// Placeholder are; TaskList/TaskItem/Mention are `Node`s.
export function sharedExtensions(opts: {
  placeholder: string;
  mentions?: MentionConfig[];
  /** Per-surface StarterKit overrides — e.g. Pages passes { document: false }
   *  to register its own attr-carrying Document. Absent → bare StarterKit,
   *  keeping the tasks card-editor byte-identical. */
  starterKit?: Partial<StarterKitOptions>;
}): AnyExtension[] {
  return [
    opts.starterKit ? StarterKit.configure(opts.starterKit) : StarterKit,
    Placeholder.configure({ placeholder: opts.placeholder }),
    TaskList,
    TaskItem.configure({ nested: true }),
    ...(opts.mentions ?? []).map(makeMention),
  ];
}

export { makeMention } from "./mention";
export type { MentionConfig } from "./mention";
export type { MentionItem } from "./mention-list";
export { MentionList } from "./mention-list";
