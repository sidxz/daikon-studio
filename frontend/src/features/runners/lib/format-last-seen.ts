/** Coarse and human, never a timestamp a scientist has to do math on. */
export function formatLastSeen(iso: string | null, now: number = Date.now()): string {
  if (!iso) return "Never";
  const seconds = Math.round((now - new Date(iso).getTime()) / 1000);
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
