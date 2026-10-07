import { cn } from "@/shared/lib/utils";

/**
 * Every number this product renders goes through here.
 *
 * A predicted IC50 has to read exactly like a measured one, which means the
 * unit and the direction travel with the value rather than being reattached at
 * each call site. During the backend build that pairing was dropped at four
 * separate boundaries and fixed four times; one component is the structural
 * answer, because there is then only one place that can drop it.
 *
 * Absence renders as absence. XGBoost reports no uncertainty and it comes back
 * null; applicability is null when it cannot be computed, never 0.0. Both show
 * N/A. A fabricated zero would be worse than no number at all.
 */
export function ReadoutValue({
  value,
  unit,
  direction,
  precision = 3,
  className,
  showDirection = false,
}: {
  value: number | null | undefined;
  unit?: string | null;
  direction?: string | null;
  precision?: number;
  className?: string;
  showDirection?: boolean;
}) {
  if (value == null || Number.isNaN(value)) {
    return <span className={cn("text-muted-foreground", className)}>N/A</span>;
  }

  const magnitude = Math.abs(value);
  const formatted =
    magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)
      ? value.toExponential(2)
      : value.toFixed(precision);

  return (
    <span className={cn("tabular-nums", className)}>
      {formatted}
      {unit && <span className="ml-1 text-muted-foreground">{unit}</span>}
      {showDirection && direction && (
        <span className="ml-1 text-xs text-muted-foreground">({direction} is better)</span>
      )}
    </span>
  );
}
