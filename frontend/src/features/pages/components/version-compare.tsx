"use client";

import { useMemo, useState } from "react";

import { usePageContent } from "@/features/pages/lib/hooks";
import { diffDocs } from "@/features/pages/lib/pm-diff";
import type { PMDoc, RevisionView } from "@/features/pages/lib/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";

import { PageView } from "./page-view";

/** Split compare: two version pickers, two read-only panes in ONE scroll
 *  container. Left highlights what the older version loses (red), right what
 *  the newer one gains (green) — each pane's ranges are in its own doc, so
 *  there is no cross-pane position mapping. ponytail: no scroll-sync; the
 *  shared scroller is enough for similar-length versions — add anchor-based
 *  sync if drift annoys. */
export function VersionCompare({
  pageId,
  revisions,
  initialLeftNo,
  initialRightNo,
}: {
  pageId: string;
  revisions: RevisionView[]; // ascending revision_no
  initialLeftNo: number;
  initialRightNo: number;
}) {
  const [leftNo, setLeftNo] = useState(initialLeftNo);
  const [rightNo, setRightNo] = useState(initialRightNo);

  const bySide = (no: number) => revisions.find((r) => r.revision_no === no) ?? null;
  const left = bySide(leftNo);
  const right = bySide(rightNo);
  const { data: leftDoc } = usePageContent(pageId, left?.sha256 ?? null);
  const { data: rightDoc } = usePageContent(pageId, right?.sha256 ?? null);

  const diff = useMemo(
    () => (leftDoc && rightDoc ? diffDocs(leftDoc as PMDoc, rightDoc as PMDoc) : null),
    [leftDoc, rightDoc],
  );

  const ordinal = (no: number) => revisions.findIndex((r) => r.revision_no === no) + 1;
  const picker = (value: number, onChange: (no: number) => void) => (
    <Select value={String(value)} onValueChange={(v) => onChange(Number(v))}>
      <SelectTrigger className="w-40">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {revisions.map((r) => (
          <SelectItem key={r.revision_no} value={String(r.revision_no)}>
            Version {ordinal(r.revision_no)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        {picker(leftNo, setLeftNo)}
        <span className="text-sm text-muted-foreground">compared with</span>
        {picker(rightNo, setRightNo)}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {diff && left && right ? (
          <div className="grid grid-cols-2 gap-4 px-4">
            <div className="min-w-0 border-r border-border pr-4">
              {/* Key includes BOTH shas: this pane's decorations are computed from
                  diffDocs(left, right), so changing either picker must remount it,
                  not just the pane whose own sha changed. */}
              <PageView
                key={`l-${left.sha256}-${right.sha256}`}
                content={(leftDoc as PMDoc) ?? null}
                diff={{ ranges: diff.left, side: "del" }}
              />
            </div>
            <div className="min-w-0">
              <PageView
                key={`r-${left.sha256}-${right.sha256}`}
                content={(rightDoc as PMDoc) ?? null}
                diff={{ ranges: diff.right, side: "ins" }}
              />
            </div>
          </div>
        ) : (
          <div className="p-6 text-muted-foreground">Loading…</div>
        )}
      </div>
    </div>
  );
}
