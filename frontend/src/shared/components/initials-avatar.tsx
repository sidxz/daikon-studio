import { Avatar, AvatarFallback } from "@/shared/components/ui/avatar";

/** First letters of the first and last words: "Ada Lovelace" is AL, "Ada" is A. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/);
  const last = words.length > 1 ? words[words.length - 1] : "";
  return `${words[0]?.[0] ?? ""}${last[0] ?? ""}`.toUpperCase();
}

/** A member's initials in a small circle, named for hover and screen readers. Nothing when unknown. */
export function InitialsAvatar({ name }: { name: string | undefined }) {
  if (!name) return null;
  return (
    <Avatar role="img" aria-label={name} title={name} className="size-5">
      <AvatarFallback aria-hidden className="text-[10px] font-medium">
        {initials(name)}
      </AvatarFallback>
    </Avatar>
  );
}
