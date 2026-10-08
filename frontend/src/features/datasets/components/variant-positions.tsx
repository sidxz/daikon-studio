"use client";

import { RESIDUE_AXIS } from "@/features/datasets/lib/residues";
import type { DatasetProfile } from "@/features/datasets/types";
import { ChartLegend, PlotFigure, baseOptions, useChartTheme } from "@/shared/components/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import * as Plot from "@observablehq/plot";
import { useCallback } from "react";

type Bar = { position: number; partition: string; count: number };
type Cell = {
  position: number;
  variant: string;
  value: number;
  split: string;
  label: string;
};

const PARTITIONS = ["train", "validation", "test"] as const;

/**
 * Where a single-parent variant series varies, along the protein.
 *
 * This is the sequence replacement for the chemical-space map, which is suppressed for
 * sequence datasets because a Tanimoto layout of protein is a convincing picture of
 * nothing. The honest equivalent is not a map of similarity but a map of *position*:
 * the axis a variant series actually moves along, and the axis a position split holds
 * out. A column appearing in two colors is the leakage the split exists to prevent, so
 * the integrity of the split is legible here rather than only assertable.
 */
export function VariantPositions({ profile }: { profile: DatasetProfile }) {
  const theme = useChartTheme();
  const variants = profile.variants;

  const bars: Bar[] = (variants?.positions ?? []).flatMap((entry) =>
    PARTITIONS.filter((partition) => entry[partition] > 0).map((partition) => ({
      position: entry.position,
      partition,
      count: entry[partition],
    })),
  );

  const options = useCallback((): Plot.PlotOptions => {
    if (!theme) return {};
    return {
      ...baseOptions(theme),
      ariaLabel: "Variants per residue position, colored by partition",
      x: {
        label: "residue position →",
        grid: true,
        // The whole protein, not just the varying stretch: a series that probes forty
        // residues of a 286-residue protein should look like forty residues of a
        // 286-residue protein.
        domain: [1, variants?.consensus.length ?? 1],
      },
      y: { label: "variants", grid: true, zero: true },
      color: {
        domain: [...PARTITIONS],
        range: [...theme.triple],
      },
      marks: [
        Plot.ruleY([0], { stroke: theme.grid }),
        Plot.rectY(bars, {
          x: "position",
          y: "count",
          fill: "partition",
          interval: 1,
          title: (bar: Bar) =>
            `position ${bar.position} · ${bar.count} variant${bar.count === 1 ? "" : "s"} in ${bar.partition}`,
        }),
      ],
    };
  }, [theme, bars, variants]);

  if (!variants || variants.positions.length === 0) return null;

  const held = variants.held_out_positions;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Where this series varies</CardTitle>
        <p className="text-sm text-muted-foreground">
          {variants.positions.length} of {variants.consensus.length} residue positions carry a
          variant.{" "}
          {held > 0
            ? `${held} ${held === 1 ? "position is" : "positions are"} absent from training, which is what the test score is measured on.`
            : "Every position with a variant also appears in training, so the test set asks nothing new of the model."}
          {variants.multi_mutant_rows > 0 &&
            ` ${variants.multi_mutant_rows} rows change more than one position.`}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <PlotFigure options={options} height={200} />
        <ChartLegend
          items={PARTITIONS.map((partition, index) => ({
            label: partition,
            color: theme?.triple[index] ?? "currentColor",
          }))}
        />
      </CardContent>
    </Card>
  );
}

/**
 * The mutational map: every measured substitution, position by replacement residue.
 *
 * This is how the field reads a variant experiment, and it is the real counterpart of
 * the molecule grid a chemist scans -- not a per-row picture (a variant's sequence is
 * identical to its parent for hundreds of characters, so drawing it bigger shows
 * nothing) but the whole experiment at once. A row of dark cells is a position that
 * cannot tolerate substitution; a gap is a position nobody measured.
 *
 * The y axis is ordered by property rather than alphabetically, so chemically similar
 * replacements sit together and a band of colour means something.
 */
export function SubstitutionMap({ profile }: { profile: DatasetProfile }) {
  const theme = useChartTheme();
  const variants = profile.variants;
  const substitutions = variants?.substitutions ?? [];

  const cells: Cell[] = substitutions.map((sub) => ({
    position: sub.position,
    variant: sub.variant,
    value: sub.value,
    split: sub.split,
    label: `${sub.wild_type}${sub.position}${sub.variant} · ${sub.value.toFixed(3)} · ${sub.split}`,
  }));

  const options = useCallback((): Plot.PlotOptions => {
    if (!theme) return {};
    return {
      ...baseOptions(theme),
      ariaLabel: "Measured value for every substitution, by position and replacement residue",
      marginLeft: 58,
      x: { label: "residue position →", type: "band", tickFormat: (d: number) => String(d) },
      y: {
        label: "replaced by",
        domain: [...RESIDUE_AXIS],
        tickSize: 0,
        // Every residue, not Plot's thinned subset: the axis is a 20-item alphabet, and
        // a skipped letter makes the row below it unreadable rather than merely sparse.
        ticks: [...RESIDUE_AXIS],
      },
      color: {
        type: "diverging",
        scheme: "RdBu",
        label: "measured value",
        legend: true,
      },
      marks: [
        Plot.cell(cells, {
          x: "position",
          y: "variant",
          fill: "value",
          inset: 0.3,
          title: "label",
        }),
      ],
    };
  }, [theme, cells]);

  if (!variants || cells.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Every measured substitution</CardTitle>
        <p className="text-sm text-muted-foreground">
          {cells.length.toLocaleString()} substitutions across {variants.positions.length}{" "}
          positions. A dark column is a position where most replacements changed the measurement; a
          gap is a substitution nobody made.
          {variants.positions_sampled_from != null &&
            ` Showing the ${variants.positions.length} most-varied of ${variants.positions_sampled_from} positions.`}
        </p>
      </CardHeader>
      <CardContent>
        {/* Height is a PlotFigure prop, not a Plot option -- the component applies it
            after spreading options, so setting it there is silently dropped. Twenty
            residue rows need the room: at the 220 default the axis is a grey smear and
            the map loses the half of its meaning that says what the residue became. */}
        <PlotFigure options={options} height={460} />
      </CardContent>
    </Card>
  );
}
