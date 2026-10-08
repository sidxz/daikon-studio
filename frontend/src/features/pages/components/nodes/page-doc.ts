import { Document } from "@tiptap/extension-document";

import { PAGE_SETTINGS_DEFAULTS } from "@/features/pages/lib/page-settings";

/** Pages' document node: per-page settings are doc attributes, so getJSON()
 *  round-trips them through the content-addressed body with zero backend
 *  involvement. Registered alongside sharedExtensions({ starterKit:
 *  { document: false } }) — exactly one Document node in the editor. */
export const PageDoc = Document.extend({
  addAttributes() {
    return {
      layout: { default: PAGE_SETTINGS_DEFAULTS.layout },
      textSize: { default: PAGE_SETTINGS_DEFAULTS.textSize },
      locked: { default: PAGE_SETTINGS_DEFAULTS.locked },
    };
  },
});
