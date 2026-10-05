/** "Oct 5", with the year only when it is not the current one. `options` adds fields, like a weekday. */
export function shortDate(
  iso: string,
  now: Date = new Date(),
  options: Intl.DateTimeFormatOptions = {},
): string {
  const date = new Date(iso);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === now.getFullYear() ? undefined : "numeric",
    ...options,
  });
}
