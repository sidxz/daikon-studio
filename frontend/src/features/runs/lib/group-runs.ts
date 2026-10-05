export interface RunDayGroup<T> {
  day: string;
  runs: T[];
}

/** Calendar day, keeping the incoming order (newest first), so each day lists newest first. */
export function groupRunsByDay<T extends { created_at: string }>(
  runs: T[],
  dayOf: (iso: string) => string = (iso) =>
    new Date(iso).toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
): RunDayGroup<T>[] {
  const days = new Map<string, T[]>();
  for (const run of runs) {
    const day = dayOf(run.created_at);
    days.set(day, [...(days.get(day) ?? []), run]);
  }
  return [...days].map(([day, dayRuns]) => ({ day, runs: dayRuns }));
}
