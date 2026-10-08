"use client";

import type { DatasetProfile } from "@/features/datasets/types";
import { ChartLegend, PlotFigure, baseOptions, useChartTheme } from "@/shared/components/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import * as Plot from "@observablehq/plot";
import { useCallback } from "react";

type Bar = { position: number; partition: string; count: number };

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
      height: 200,
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
        <PlotFigure options={options} />
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
