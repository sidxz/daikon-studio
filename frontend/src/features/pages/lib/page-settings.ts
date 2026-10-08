import type { PMDoc } from "./types";

/** Per-page settings that ride the ProseMirror doc's own attributes — inside
 *  the content-addressed body, so Save persists them, history versions them,
 *  and Restore restores them. See the 2026-07-22 settings design spec. */
export type PageSettings = {
  layout: "standard" | "full";
  textSize: "default" | "small";
  locked: boolean;
};

export const PAGE_SETTINGS_DEFAULTS: PageSettings = {
  layout: "standard",
  textSize: "default",
  locked: false,
};

/** Tolerant resolver: docs saved before settings existed have no attrs, and a
 *  future writer could hold junk — anything unrecognized becomes the default. */
export function pageSettings(doc: PMDoc | null | undefined): PageSettings {
  const a = doc?.attrs ?? {};
  return {
    layout: a.layout === "full" ? "full" : "standard",
    textSize: a.textSize === "small" ? "small" : "default",
    locked: a.locked === true,
  };
}

/** Wrapper classes for EditorContent: reading-column width + small-text hook
 *  (`.tiptap-small` rules live in globals.css). */
export function pageSettingsClasses(s: PageSettings): string {
  return [
    s.layout === "standard" ? "mx-auto w-full max-w-4xl" : "w-full",
    s.textSize === "small" ? "tiptap-small" : "",
  ]
    .filter(Boolean)
    .join(" ");
}
