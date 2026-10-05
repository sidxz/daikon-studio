import { InitialsAvatar } from "@/shared/components/initials-avatar";
import { Badge } from "@/shared/components/ui/badge";
import { shortDate } from "@/shared/lib/format-date";
import { cn } from "@/shared/lib/utils";
import Link from "next/link";
import type { DragEventHandler, ReactNode } from "react";

/**
 * A dataset or protocol in its list: name, one lead number, and who made it when.
 * The title link is stretched over the card, so `action` is its sibling, never
 * nested in the <a>. A draft's edge is dashed, as in the logo: solid is
 * measured, dashed is provisional.
 */
export function ItemCard({
  href,
  name,
  subtitle,
  draft = false,
  action,
  footerStart,
  creator,
  createdAt,
  draggable,
  onDragStart,
  children,
}: {
  href: string;
  name: string;
  subtitle?: ReactNode;
  draft?: boolean;
  /** Shown on hover or focus, and always on touch. */
  action?: ReactNode;
  footerStart?: ReactNode;
  creator?: string;
  createdAt: string;
  draggable?: boolean;
  onDragStart?: DragEventHandler<HTMLDivElement>;
  children?: ReactNode;
}) {
  return (
    <div
      draggable={draggable}
      onDragStart={onDragStart}
      className={cn(
        "group relative flex h-full flex-col gap-4 rounded-lg border border-border p-4 transition-colors hover:border-foreground/20",
        draft && "border-dashed",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <Link
            href={href}
            draggable={false}
            className="font-medium outline-none after:absolute after:inset-0 after:rounded-lg focus-visible:after:ring-2 focus-visible:after:ring-ring"
          >
            {name}
          </Link>
          {subtitle && <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>}
        </div>
        <div className="relative z-10 flex h-7 shrink-0 items-center gap-1">
          {draft && (
            <Badge variant="outline" className="font-normal text-muted-foreground">
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

      {children}

      <div className="mt-auto flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="min-w-0 truncate">{footerStart}</span>
        <span className="flex shrink-0 items-center gap-2">
          <InitialsAvatar name={creator} />
          {shortDate(createdAt)}
        </span>
      </div>
    </div>
  );
}

/** The card's one bold element: a label over a large number, with an optional muted note beside it. */
export function LeadNumber({
  label,
  value,
  aside,
}: {
  label: ReactNode;
  value: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="flex items-baseline-last gap-3">
        <span className="text-2xl font-semibold tabular-nums">{value}</span>
        {aside && <p className="text-xs leading-4 text-muted-foreground">{aside}</p>}
      </div>
    </div>
  );
}
