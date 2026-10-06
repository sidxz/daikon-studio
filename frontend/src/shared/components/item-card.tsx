import { InitialsAvatar } from "@/shared/components/initials-avatar";
import { Badge } from "@/shared/components/ui/badge";
import { shortDate } from "@/shared/lib/format-date";
import { cn } from "@/shared/lib/utils";
import Link from "next/link";
import type { DragEventHandler, ReactNode } from "react";

/** A compact research card. The stretched title link stays separate from its actions. */
export function ItemCard({
  href,
  name,
  subtitle,
  draft = false,
  action,
  footerStart,
  creator,
  creatorId,
  createdAt,
  draggable,
  onDragStart,
  compact = false,
  children,
}: {
  href: string;
  name: string;
  subtitle?: ReactNode;
  draft?: boolean;
  action?: ReactNode;
  footerStart?: ReactNode;
  creator?: string;
  creatorId?: string | null;
  createdAt: string;
  draggable?: boolean;
  onDragStart?: DragEventHandler<HTMLDivElement>;
  compact?: boolean;
  children?: ReactNode;
}) {
  return (
    <div
      draggable={draggable}
      onDragStart={onDragStart}
      className={cn(
        "group relative flex h-full flex-col rounded-md border border-border bg-card transition-colors hover:border-primary/45",
        compact ? "gap-2.5 p-3.5" : "gap-3 p-4",
        draft && "border-dashed",
      )}
    >
      <div>
        <div className="flex items-start justify-between gap-2">
          <Link
            href={href}
            draggable={false}
            className={cn(
              "min-w-0 break-words outline-none after:absolute after:inset-0 after:rounded-md focus-visible:after:ring-2 focus-visible:after:ring-ring",
              compact
                ? "text-[15px] font-semibold leading-5"
                : "text-sm font-medium leading-relaxed",
            )}
          >
            {name}
          </Link>
          <div className="relative z-10 flex h-5 shrink-0 items-center gap-1">
            {draft && (
              <Badge
                variant="outline"
                className="border-dashed text-[10px] font-normal text-muted-foreground"
              >
                Draft
              </Badge>
            )}
            {action && (
              <div className="opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 has-[[data-state=open]]:opacity-100 [@media(hover:none)]:opacity-100">
                {action}
              </div>
            )}
          </div>
        </div>
        {subtitle && <p className="mt-1 truncate text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
      <div
        className={cn(
          "mt-auto flex items-center justify-between gap-3 border-t border-border/60 pt-2 text-muted-foreground",
          compact ? "text-[11px]" : "text-xs",
        )}
      >
        <span className="min-w-0 truncate">{footerStart}</span>
        <span className="flex shrink-0 items-center gap-2">
          <InitialsAvatar name={creator} id={creatorId} />
          {shortDate(createdAt)}
        </span>
      </div>
    </div>
  );
}

export function LeadNumber({
  label,
  value,
  aside,
}: { label: ReactNode; value: ReactNode; aside?: ReactNode }) {
  return (
    <div>
      <p className="truncate text-xs text-muted-foreground">{label}</p>
      <div className="flex items-baseline-last gap-3">
        <span className="text-xl font-semibold tabular-nums">{value}</span>
        {aside && <p className="text-xs leading-4 text-muted-foreground">{aside}</p>}
      </div>
    </div>
  );
}
