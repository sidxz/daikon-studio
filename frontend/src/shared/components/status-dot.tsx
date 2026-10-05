import { cn } from "@/shared/lib/utils";

export type StatusTone = "success" | "active" | "failed" | "muted";

const TONE: Record<StatusTone, string> = {
  success: "bg-success",
  // The page's one ambient motion: work is happening now. Static under reduced motion.
  active: "bg-icon-runs motion-safe:animate-status-breathe",
  failed: "bg-destructive",
  muted: "bg-muted-foreground",
};

/** A small status dot. Decorative: the word beside it carries the meaning. */
export function StatusDot({ tone }: { tone: StatusTone }) {
  return (
    <span aria-hidden className={cn("inline-block size-1.5 shrink-0 rounded-full", TONE[tone])} />
  );
}
