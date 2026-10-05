/** One target's headline on a training run (`Run.record_metrics`). */
export interface Headline {
  column: string;
  primary_metric: string;
  value: number | null;
  baseline_value: number | null;
}

/** A run's per-target headlines; empty while it has none. `metrics` is untyped in the contract. */
export function headlines(metrics: unknown): Headline[] {
  const targets = (metrics as { targets?: unknown } | null)?.targets;
  return Array.isArray(targets) ? (targets as Headline[]) : [];
}
