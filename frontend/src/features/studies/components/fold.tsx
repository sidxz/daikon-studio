import type { ReactNode } from "react";

/**
 * A line in the thread that opens.
 *
 * The rail's height is the conversation's, so what is worth keeping but not worth reading
 * every time -- what a turn left unsettled, where the study stands -- costs one compact line
 * until it is asked for. `<details>` rather than a component: the open state survives a
 * re-render without being held anywhere, and the keyboard and find-in-page already work.
 */
export function Fold({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="rounded-sm border border-dashed px-2 py-1">
      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
        {label}
      </summary>
      <div className="mt-1 text-sm text-muted-foreground">{children}</div>
    </details>
  );
}
