/**
 * Ordering for the scorecard's "Largest prediction errors" grid.
 *
 * A series is two or more of these compounds sharing a Bemis–Murcko scaffold:
 * a cluster of misses there points to a chemical series the model predicts
 * poorly. Compounds with no ring system share the empty scaffold but no core,
 * so they never form a series.
 */

export type WorstOrder = "series" | "error";

export interface SeriesTag {
  /** "A" for the largest series, then "B", ... */
  label: string;
  size: number;
  scaffold: string;
  /** Position in series order, for picking a color. */
  index: number;
}

interface WorstRowLike {
  scaffold: string;
  residual: number;
}

const worstFirst = (a: WorstRowLike, b: WorstRowLike) =>
  Math.abs(b.residual) - Math.abs(a.residual);

/**
 * By series: each series' compounds side by side, the largest series first (a
 * tie goes to the series with the worse miss), each block worst first; then
 * every other compound, worst first. By error: worst first throughout, with the
 * same series tags, so a cluster shows as repeated chips.
 */
export function orderWorstRows<T extends WorstRowLike>(
  rows: readonly T[],
  order: WorstOrder,
): { row: T; series: SeriesTag | null }[] {
  const byScaffold = new Map<string, T[]>();
  for (const row of rows) {
    if (!row.scaffold) continue;
    byScaffold.set(row.scaffold, [...(byScaffold.get(row.scaffold) ?? []), row]);
  }
  const series = [...byScaffold.values()]
    .filter((members) => members.length > 1)
    .map((members) => [...members].sort(worstFirst))
    .sort((a, b) => b.length - a.length || worstFirst(a[0], b[0]));

  const tagOf = new Map<T, SeriesTag>();
  series.forEach((members, index) => {
    const tag = {
      label: String.fromCharCode(65 + index),
      size: members.length,
      scaffold: members[0].scaffold,
      index,
    };
    for (const row of members) tagOf.set(row, tag);
  });

  const ordered =
    order === "series"
      ? [...series.flat(), ...rows.filter((row) => !tagOf.has(row)).sort(worstFirst)]
      : [...rows].sort(worstFirst);
  return ordered.map((row) => ({ row, series: tagOf.get(row) ?? null }));
}
