/** Swatches drawn the way the map draws each kind of point. */
export type Swatch = "train" | "validation" | "test" | "runIn" | "runOut";

function SwatchIcon({ kind }: { kind: Swatch }) {
  return (
    <svg viewBox="0 0 12 12" className="size-3 shrink-0" aria-hidden="true">
      {kind === "train" && <circle cx={6} cy={6} r={3} className="fill-chart-1" />}
      {kind === "validation" && <circle cx={6} cy={6} r={3} className="fill-chart-2" />}
      {kind === "test" && <circle cx={6} cy={6} r={3} className="fill-score-fair" />}
      {kind === "runIn" && <circle cx={6} cy={6} r={4.5} className="fill-score-fair" />}
      {kind === "runOut" && (
        <circle
          cx={6}
          cy={6}
          r={4}
          className="fill-none stroke-score-fair"
          strokeWidth={1.5}
          strokeDasharray="2 1.6"
        />
      )}
    </svg>
  );
}

export function MapLegend({ items }: { items: { swatch: Swatch; label: string }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
      {items.map((item) => (
        <li key={item.swatch} className="inline-flex items-center gap-1.5">
          <SwatchIcon kind={item.swatch} />
          {item.label}
        </li>
      ))}
    </ul>
  );
}
