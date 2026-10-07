import { formatCutoff } from "@/features/protocols";
import { ReadoutValue } from "@/shared/components/readout-value";
import { cn } from "@/shared/lib/utils";
import { formatThousandths } from "../lib/cell-scale";
import { IN_DOMAIN_FLOOR } from "../lib/result-query";

/**
 * The triage grid's cells: each value stays a number, with a thin bar under it so a
 * column can be read at a glance. The bar is decoration for sighted scanning
 * (`aria-hidden`); the number, and the label where there is one, carry the meaning.
 * Colours come from the design tokens and were checked for contrast and colour-vision
 * separation in both themes: the accent for a value, score-fair amber for
 * uncertainty, and the foreground at half strength for "below the cutoff" and "in
 * domain".
 */

function Meter({
  fraction,
  fill,
  tick,
  outside = false,
}: {
  fraction: number;
  fill: string;
  /** A mark across the track at this fraction: a cutoff or a floor. */
  tick?: number;
  /**
   * Hatched rather than solid, so "outside the domain" does not rest on colour
   * alone; the cell also says it in words.
   */
  outside?: boolean;
}) {
  return (
    <div aria-hidden className="relative h-1.5 w-full max-w-28 rounded-full bg-foreground/10">
      <div
        className={cn("h-full rounded-full", !outside && fill)}
        style={{
          width: `${fraction * 100}%`,
          backgroundImage: outside
            ? "repeating-linear-gradient(135deg, var(--color-warning) 0 2px, transparent 2px 4px)"
            : undefined,
        }}
      />
      {tick != null && (
        <div
          className="absolute -top-1 h-3.5 w-0.5 rounded-full bg-foreground/70"
          style={{ left: `calc(${tick * 100}% - 1px)` }}
        />
      )}
    </div>
  );
}

function Stack({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex h-full w-full flex-col justify-center gap-1.5" title={title}>
      {children}
    </div>
  );
}

const Absent = () => <span className="text-muted-foreground">N/A</span>;

/** P(class=1). A positive class does not imply a desirable compound. */
export function ProbabilityCell({ value, cutoff }: { value: number | null; cutoff: number }) {
  if (value == null) return <Absent />;
  return (
    <Stack
      title={`Probability of class 1: ${value.toPrecision(3)}. Predicted positive at ${formatCutoff(cutoff)} or above.`}
    >
      <span className="tabular-nums">{formatThousandths(value)}</span>
      <Meter fraction={value} fill="bg-primary" tick={cutoff} />
    </Stack>
  );
}

/** A predicted class as a word: "1.000" and "0.000" make a chemist decode a label. */
export function ClassCell({ value }: { value: number | null }) {
  if (value == null) return <Absent />;
  const active = value >= 0.5;
  return (
    <span
      title={active ? "Positive (class 1)" : "Negative (class 0)"}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium",
        active ? "bg-primary/10 text-foreground" : "border border-border text-muted-foreground",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "size-1.5 rounded-full",
          active ? "bg-primary" : "border border-muted-foreground",
        )}
      />
      {active ? "Positive" : "Negative"}
    </span>
  );
}

/** A continuous prediction on this run's own range, lowest to highest. */
export function ValueCell({
  value,
  unit,
  fraction,
  min,
  max,
}: {
  value: number | null;
  unit: string | null | undefined;
  fraction: number | null;
  min: number | null | undefined;
  max: number | null | undefined;
}) {
  if (value == null) return <Absent />;
  return (
    <Stack
      title={
        min != null && max != null
          ? `${value.toPrecision(4)}${unit ? ` ${unit}` : ""}. This run ranges from ${min.toPrecision(3)} to ${max.toPrecision(3)}.`
          : `${value.toPrecision(4)}${unit ? ` ${unit}` : ""}`
      }
    >
      <ReadoutValue value={value} unit={unit} precision={3} />
      {fraction != null && <Meter fraction={fraction} fill="bg-primary" />}
    </Stack>
  );
}

/** How unsure the model is; amber draws the eye to the shakiest predictions. */
export function UncertaintyCell({ value, scale }: { value: number | null; scale: number | null }) {
  // Null for an engine with nothing to report: absence, never a fabricated zero.
  if (value == null) return <Absent />;
  return (
    <Stack
      title={`Uncertainty ${value.toPrecision(3)}${scale != null ? `. Bar scale 0 to ${scale.toPrecision(2)}.` : "."}`}
    >
      <span className="tabular-nums">{formatThousandths(value)}</span>
      {scale != null && scale > 0 && (
        <Meter fraction={Math.min(value / scale, 1)} fill="bg-score-fair" />
      )}
    </Stack>
  );
}

/** Tanimoto similarity to the nearest training compound, with the domain floor marked. */
export function ApplicabilityCell({ value }: { value: number | null }) {
  if (value == null) return <Absent />;
  const outside = value < IN_DOMAIN_FLOOR;
  const percent = `${(value * 100).toFixed(0)}%`;
  return (
    <Stack
      title={`Tanimoto similarity to nearest training compound: ${percent}. The applicability domain starts at ${(IN_DOMAIN_FLOOR * 100).toFixed(0)}%.`}
    >
      <span className={cn("tabular-nums", outside && "text-warning")}>
        {percent}
        {outside && <span className="ml-1.5 text-xs font-medium">Outside</span>}
      </span>
      <Meter
        fraction={value}
        // Lighter than the readouts' bars: applicability qualifies a prediction, it
        // is not one, and a run of 100% rows should not outweigh them.
        fill={outside ? "bg-warning" : "bg-foreground/30"}
        tick={IN_DOMAIN_FLOOR}
        outside={outside}
      />
    </Stack>
  );
}
