import { shortDate } from "@/shared/lib/format-date";

export interface RunDayGroup<T> {
  day: string;
  runs: T[];
}

/** Local midnight, as a timestamp. */
const startOfDay = (date: Date) =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

/** "Today", "Yesterday", else "Sat, Oct 3", with the year only when it is not this one. */
export function dayLabel(iso: string, now: Date): string {
  // Rounded, so a 23- or 25-hour day across a clock change still counts as one.
  const daysAgo = Math.round((startOfDay(now) - startOfDay(new Date(iso))) / 86_400_000);
  if (daysAgo === 0) return "Today";
  if (daysAgo === 1) return "Yesterday";
  return shortDate(iso, now, { weekday: "short" });
}

/** Calendar day, keeping the incoming order (newest first), so each day lists newest first. */
export function groupRunsByDay<T extends { created_at: string }>(
  runs: T[],
  dayOf: (iso: string) => string,
): RunDayGroup<T>[] {
  const days = new Map<string, T[]>();
  for (const run of runs) {
    const day = dayOf(run.created_at);
    days.set(day, [...(days.get(day) ?? []), run]);
  }
  return [...days].map(([day, dayRuns]) => ({ day, runs: dayRuns }));
}
