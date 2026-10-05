/** The fields grouping reads; the full `Run` satisfies it. */
interface GroupableRun {
  protocol_id?: string | null;
  created_at: string;
}

export interface RunDayGroup<T> {
  day: string;
  runs: T[];
}

export interface RunProtocolGroup<T> {
  protocolId: string | null;
  days: RunDayGroup<T>[];
}

/**
 * Protocol, then calendar day, keeping the incoming order (newest first), so
 * the protocol with the latest run leads and each day lists newest first.
 */
export function groupRuns<T extends GroupableRun>(
  runs: T[],
  dayOf: (iso: string) => string = (iso) =>
    new Date(iso).toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
): RunProtocolGroup<T>[] {
  const groups = new Map<string | null, Map<string, T[]>>();
  for (const run of runs) {
    const protocolId = run.protocol_id ?? null;
    const days = groups.get(protocolId) ?? new Map<string, T[]>();
    groups.set(protocolId, days);
    const day = dayOf(run.created_at);
    days.set(day, [...(days.get(day) ?? []), run]);
  }
  return [...groups].map(([protocolId, days]) => ({
    protocolId,
    days: [...days].map(([day, dayRuns]) => ({ day, runs: dayRuns })),
  }));
}
