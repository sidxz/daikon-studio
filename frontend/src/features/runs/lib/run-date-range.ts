export function localDay(date: Date): string {
  return `${date.getFullYear().toString().padStart(4, "0")}-${(date.getMonth() + 1).toString().padStart(2, "0")}-${date.getDate().toString().padStart(2, "0")}`;
}

export function parseLocalDay(day: string): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) return null;
  const date = new Date(`${day}T00:00:00`);
  return Number.isFinite(date.getTime()) && localDay(date) === day ? date : null;
}

/** The selected end day is inclusive; advance by calendar day, including across DST. */
export function runDateBounds(from: string, to: string) {
  const start = parseLocalDay(from);
  const end = parseLocalDay(to);
  if (end) end.setDate(end.getDate() + 1);
  return { createdFrom: start?.toISOString(), createdBefore: end?.toISOString() };
}
