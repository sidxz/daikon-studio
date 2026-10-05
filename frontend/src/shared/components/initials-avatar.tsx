import { Avatar, AvatarFallback } from "@/shared/components/ui/avatar";
import { cn } from "@/shared/lib/utils";

/** First letters of the first and last words: "Ada Lovelace" is AL, "Ada" is A. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/);
  const last = words.length > 1 ? words[words.length - 1] : "";
  return `${words[0]?.[0] ?? ""}${last[0] ?? ""}`.toUpperCase();
}

// Six existing hues, written out so Tailwind sees each class. The text is the hue
// mixed toward the foreground, so it reads on its own tint in both themes. Ordered
// so neighbors are far apart in hue (red and pink are not adjacent): keys that
// differ by one character land on neighboring slots.
const TINTS = [
  "bg-chart-1/15 text-[color:color-mix(in_oklab,var(--chart-1)_60%,var(--foreground))]",
  "bg-chart-4/15 text-[color:color-mix(in_oklab,var(--chart-4)_60%,var(--foreground))]",
  "bg-chart-2/15 text-[color:color-mix(in_oklab,var(--chart-2)_60%,var(--foreground))]",
  "bg-chart-5/15 text-[color:color-mix(in_oklab,var(--chart-5)_60%,var(--foreground))]",
  "bg-chart-3/15 text-[color:color-mix(in_oklab,var(--chart-3)_60%,var(--foreground))]",
  "bg-icon-collections/15 text-[color:color-mix(in_oklab,var(--icon-collections)_60%,var(--foreground))]",
];

/**
 * The same tint for the same user everywhere, so two people who share initials
 * ("Sid Rath", "Siddhant Rath") still look different. ponytail: six hues, so one
 * pair in six still collides; the name is always in the title.
 */
export function tintFor(key: string): string {
  let hash = 0;
  for (const char of key) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return TINTS[hash % TINTS.length];
}

/** A member's initials in a small tinted circle, named for hover and screen readers. Nothing when unknown. */
export function InitialsAvatar({
  name,
  id,
}: {
  name: string | undefined;
  /** The user id the tint is keyed on; the name stands in without one. */
  id?: string | null;
}) {
  if (!name) return null;
  return (
    <Avatar role="img" aria-label={name} title={name} className="size-5">
      <AvatarFallback aria-hidden className={cn("text-[10px] font-medium", tintFor(id || name))}>
        {initials(name)}
      </AvatarFallback>
    </Avatar>
  );
}
