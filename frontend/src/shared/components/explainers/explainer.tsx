"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/shared/components/ui/collapsible";
import { ChevronRight, RotateCcw } from "lucide-react";
import { type ReactNode, useEffect, useRef } from "react";
import { useExplainerStore } from "./explainer-store";
import { useTimeline } from "./use-timeline";

interface ExplainerProps {
  /** Persistence key, one per figure: closing it once closes it everywhere it appears. */
  id: string;
  label?: string;
  caption: ReactNode;
  durationMs: number;
  /** Replays the figure when this changes, e.g. the selected split strategy. */
  replayKey?: string;
  children: (t: number) => ReactNode;
}

export function Explainer({
  id,
  label = "How this works",
  caption,
  durationMs,
  replayKey,
  children,
}: ExplainerProps) {
  const open = useExplainerStore((state) => !state.closed[id]);
  const setOpen = useExplainerStore((state) => state.setOpen);
  return (
    <Collapsible open={open} onOpenChange={(next) => setOpen(id, next)} className="space-y-3">
      <CollapsibleTrigger className="group inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline">
        <ChevronRight className="size-4 transition-transform group-data-[state=open]:rotate-90 motion-reduce:transition-none" />
        {label}
      </CollapsibleTrigger>
      {/* Radix unmounts closed content, so the timeline starts on open. */}
      <CollapsibleContent>
        <ExplainerFigure caption={caption} durationMs={durationMs} replayKey={replayKey}>
          {children}
        </ExplainerFigure>
      </CollapsibleContent>
    </Collapsible>
  );
}

function ExplainerFigure({
  caption,
  durationMs,
  replayKey,
  children,
}: Omit<ExplainerProps, "id" | "label">) {
  const { ref, t, replay } = useTimeline<HTMLDivElement>(durationMs);
  const shownKey = useRef(replayKey);
  useEffect(() => {
    if (shownKey.current === replayKey) return;
    shownKey.current = replayKey;
    replay();
  }, [replayKey, replay]);

  return (
    <div className="space-y-2 border-t pt-3">
      <div ref={ref}>{children(t)}</div>
      <div className="flex items-start justify-between gap-3">
        <p className="max-w-prose text-xs text-muted-foreground">{caption}</p>
        <button
          type="button"
          onClick={replay}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <RotateCcw className="size-3" aria-hidden="true" />
          Replay
        </button>
      </div>
    </div>
  );
}
