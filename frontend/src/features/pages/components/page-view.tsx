"use client";

import type { DiffProp } from "@/features/pages/lib/pm-diff";
import type { PMDoc } from "@/features/pages/lib/types";

import { PageEditor } from "./page-editor";

/** Read-only render of a page body. Reuses PageEditor (same extensions, so embeds
 *  render identically) with `editable={false}` — used by history/preview surfaces.
 *  `diff` overlays change highlights for the compare view. */
export function PageView({ content, diff }: { content: PMDoc | null; diff?: DiffProp }) {
  return <PageEditor initial={content} onChange={() => {}} editable={false} diff={diff} />;
}
