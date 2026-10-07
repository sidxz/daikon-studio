"use client";

import * as Plot from "@observablehq/plot";
import { useCallback } from "react";
import { ChartLegend, PlotFigure, baseOptions } from "./plot-figure";
import { type ChartTheme, useChartTheme } from "./use-chart-theme";

export interface Bins {
  edges: number[];
  counts: number[];
}

export interface SplitBins {
  edges: number[];
  train: number[];
  test: number[];
}

export interface BinPoint {
  lower: number;
  upper: number;
  count: number;
  value: number;
}

/** A skeleton-shaped hole while `useChartTheme` waits for the document. */
function Pending({ height }: { height: number }) {
  return <div className="w-full animate-pulse rounded bg-muted/40" style={{ height }} />;
}

/**
 * One binned distribution.
 *
 * `reference` draws a labelled vertical rule -- zero on a residual histogram,
 * a threshold on a similarity one. It is the whole reason these charts beat the
 * numbers they summarise: a residual histogram centred half a unit right of zero
 * is a systematically optimistic model, and no aggregate on the page says that.
 */
export function HistogramChart({
  bins,
  xLabel,
  height = 200,
  reference,
  caption,
}: {
  bins: Bins;
  xLabel: string;
  height?: number;
  reference?: { at: number; label: string };
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const data = bins.counts.map((count, index) => ({
      x1: bins.edges[index],
      x2: bins.edges[index + 1],
      count,
    }));
    return {
      ...baseOptions(theme as ChartTheme),
      // Extra headroom only when a reference rule is labelled: the label sits
      // above the plot area, and at the default margin it lands on top of
      // whichever bar happens to be tallest.
      marginTop: reference ? 32 : 22,
      x: { label: xLabel, labelAnchor: "center", grid: false },
      y: { label: "compounds", grid: true, ticks: 4 },
      marks: [
        Plot.rectY(data, {
          x1: "x1",
          x2: "x2",
          y: "count",
          fill: (theme as ChartTheme).pair[0],
          // A 1px inset leaves the 2px surface gap between neighbouring bars
          // that keeps a dense histogram from reading as one solid block.
          insetLeft: 1,
          insetRight: 1,
          title: (d: { x1: number; x2: number; count: number }) =>
            `${d.x1.toFixed(2)} – ${d.x2.toFixed(2)}\n${d.count} compounds`,
        }),
        Plot.ruleY([0], { stroke: (theme as ChartTheme).grid }),
        ...(reference
          ? [
              Plot.ruleX([reference.at], {
                stroke: (theme as ChartTheme).muted,
                strokeDasharray: "3,3",
              }),
              Plot.text([reference.label], {
                x: reference.at,
                frameAnchor: "top",
                dy: -10,
                fill: (theme as ChartTheme).muted,
                fontSize: 10,
              }),
            ]
          : []),
      ],
    };
  }, [bins, xLabel, reference, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}

/**
 * Train and test on shared bins, each normalised to its own partition.
 *
 * Normalised deliberately: a train partition is typically eight times the size
 * of its test partition, so raw counts draw the test distribution as a flat line
 * along the axis and the comparison -- the only reason both are here -- becomes
 * unreadable. The y axis says "share of partition" so nobody mistakes the
 * normalisation for a count.
 */
export function SplitHistogramChart({
  bins,
  xLabel,
  height = 200,
  series = ["train", "test"],
  yLabel = "share of partition",
  caption,
}: {
  bins: SplitBins;
  xLabel: string;
  height?: number;
  yLabel?: string;
  /** What the two series are called. The fields stay `train`/`test`; only the
   *  labels change, so the classification view can say active/inactive. */
  series?: [string, string];
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const trainTotal = bins.train.reduce((sum, n) => sum + n, 0) || 1;
    const testTotal = bins.test.reduce((sum, n) => sum + n, 0) || 1;
    const data = bins.edges.slice(0, -1).flatMap((edge, index) => [
      {
        x1: edge,
        x2: bins.edges[index + 1],
        share: bins.train[index] / trainTotal,
        count: bins.train[index],
        series: series[0],
      },
      {
        x1: edge,
        x2: bins.edges[index + 1],
        share: bins.test[index] / testTotal,
        count: bins.test[index],
        series: series[1],
      },
    ]);

    return {
      ...baseOptions(theme as ChartTheme),
      x: { label: xLabel, labelAnchor: "center" },
      y: { label: yLabel, grid: true, ticks: 4, percent: true },
      color: { domain: series, range: (theme as ChartTheme).pair },
      marks: [
        Plot.rectY(data, {
          x1: "x1",
          x2: "x2",
          y: "share",
          fill: "series",
          // Translucent so the overlap is visible as overlap; the strokes below
          // keep each distribution's own outline readable through it.
          fillOpacity: 0.4,
          insetLeft: 1,
          insetRight: 1,
          title: (d: { series: string; count: number; x1: number; x2: number }) =>
            `${d.series}\n${d.x1.toFixed(2)} – ${d.x2.toFixed(2)}\n${d.count} compounds`,
        }),
        Plot.rectY(data, {
          x1: "x1",
          x2: "x2",
          y: "share",
          stroke: "series",
          strokeWidth: 1,
          fill: "none",
          insetLeft: 1,
          insetRight: 1,
        }),
        Plot.ruleY([0], { stroke: (theme as ChartTheme).grid }),
      ],
    };
  }, [bins, xLabel, series, yLabel, theme]);

  if (!theme) return <Pending height={height} />;
  return (
    <div>
      <ChartLegend
        className="mb-1"
        items={[
          { label: series[0], color: theme.pair[0] },
          { label: series[1], color: theme.pair[1] },
        ]}
      />
      <PlotFigure options={options} height={height} caption={caption} />
    </div>
  );
}

/**
 * Predicted against measured, with the y = x line every point should sit on.
 *
 * Square by construction -- identical domains on both axes -- because a parity
 * plot drawn on unequal scales puts the identity line somewhere other than 45
 * degrees, and the entire read of the chart is "how far off that line is this
 * cloud". Points are coloured by nearest-neighbour similarity so the second
 * question, whether the far-off points are the unfamiliar ones, is answered by
 * the same picture.
 */
export function ParityChart({
  points,
  unit,
  noiseFloor,
  height = 320,
  caption,
}: {
  points: { actual: number; predicted: number; similarity?: number | null }[];
  unit?: string | null;
  noiseFloor?: number | null;
  height?: number;
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const values = points.flatMap((p) => [p.actual, p.predicted]);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = (high - low) * 0.05 || 1;
    const domain: [number, number] = [low - pad, high + pad];
    const hasSimilarity = points.some((p) => p.similarity != null);
    const axis = unit ? ` (${unit})` : "";

    return {
      ...baseOptions(theme as ChartTheme),
      aspectRatio: undefined,
      x: { label: `measured${axis}`, domain, grid: true, ticks: 5 },
      y: { label: `predicted${axis}`, domain, grid: true, ticks: 5 },
      color: hasSimilarity
        ? {
            type: "linear",
            domain: [0, 1],
            range: (theme as ChartTheme).sequential,
            label: "Tanimoto similarity to nearest training compound",
            legend: true,
          }
        : undefined,
      marks: [
        // The noise floor band, when there is one: a point inside it is as
        // close to correct as the assay itself can tell.
        ...(noiseFloor
          ? [
              Plot.areaY(
                [
                  { x: domain[0], lo: domain[0] - noiseFloor, hi: domain[0] + noiseFloor },
                  { x: domain[1], lo: domain[1] - noiseFloor, hi: domain[1] + noiseFloor },
                ],
                {
                  x: "x",
                  y1: "lo",
                  y2: "hi",
                  fill: (theme as ChartTheme).muted,
                  fillOpacity: 0.12,
                },
              ),
            ]
          : []),
        Plot.line(
          [
            { x: domain[0], y: domain[0] },
            { x: domain[1], y: domain[1] },
          ],
          { x: "x", y: "y", stroke: (theme as ChartTheme).muted, strokeDasharray: "4,4" },
        ),
        Plot.dot(points, {
          x: "actual",
          y: "predicted",
          r: 2.6,
          fill: hasSimilarity ? "similarity" : (theme as ChartTheme).pair[0],
          fillOpacity: 0.65,
          // A surface-coloured hairline ring keeps overlapping points countable
          // instead of merging into one blob.
          stroke: (theme as ChartTheme).surface,
          strokeWidth: 0.4,
          title: (d: { actual: number; predicted: number; similarity?: number | null }) =>
            `measured ${d.actual.toFixed(3)}\npredicted ${d.predicted.toFixed(3)}${
              d.similarity == null ? "" : `\nsimilarity ${d.similarity.toFixed(2)}`
            }`,
        }),
      ],
    };
  }, [points, unit, noiseFloor, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}

/**
 * A measured value per bin, with the bin's population encoded as dot area.
 *
 * The dot size is not decoration. These curves are read for their trend, and a
 * bin holding four compounds and a bin holding four hundred are otherwise drawn
 * identically -- which is how a two-point wobble gets read as a finding.
 */
export function BinnedCurveChart({
  bins,
  xLabel,
  yLabel,
  height = 220,
  diagonal = false,
  caption,
}: {
  bins: BinPoint[];
  xLabel: string;
  yLabel: string;
  height?: number;
  diagonal?: boolean;
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const data = bins.map((bin) => ({ ...bin, x: (bin.lower + bin.upper) / 2 }));
    return {
      ...baseOptions(theme as ChartTheme),
      x: { label: xLabel, grid: true, ticks: 5 },
      y: { label: yLabel, grid: true, ticks: 4 },
      r: { range: [3, 9] },
      marks: [
        ...(diagonal
          ? [
              Plot.line(
                [
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ],
                {
                  x: "x",
                  y: "y",
                  stroke: (theme as ChartTheme).muted,
                  strokeDasharray: "4,4",
                },
              ),
            ]
          : []),
        Plot.line(data, {
          x: "x",
          y: "value",
          stroke: (theme as ChartTheme).pair[0],
          strokeWidth: 2,
        }),
        Plot.dot(data, {
          x: "x",
          y: "value",
          r: "count",
          fill: (theme as ChartTheme).pair[0],
          stroke: (theme as ChartTheme).surface,
          strokeWidth: 1.5,
          title: (d: BinPoint & { x: number }) =>
            `${d.lower.toFixed(2)} – ${d.upper.toFixed(2)}\n${d.value.toFixed(3)}\n${d.count} compounds`,
        }),
      ],
    };
  }, [bins, xLabel, yLabel, diagonal, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}

/**
 * Ranked horizontal bars.
 *
 * `diverging` splits the fill on the sign of the value rather than its
 * magnitude, for a quantity where the sign is the point -- a descriptor that
 * correlates negatively with the target is as informative as one that correlates
 * positively, and a single-hue ramp would rank them as though one were "less".
 */
export function HorizontalBarChart({
  data,
  xLabel,
  height = 240,
  diverging = false,
  format,
  caption,
}: {
  data: { label: string; value: number; detail?: string }[];
  xLabel: string;
  height?: number;
  diverging?: boolean;
  format?: (value: number) => string;
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const render = format ?? ((value: number) => value.toFixed(2));
    // Headroom so the longest bar stops short of the frame. Without it that
    // bar's value label -- which sits just past its end -- runs into the
    // category labels in the left margin, or off the right edge.
    const extent = Math.max(...data.map((d) => Math.abs(d.value)), 0) * 1.18 || 1;
    return {
      ...baseOptions(theme as ChartTheme),
      marginLeft: 140,
      marginRight: 44,
      x: {
        label: xLabel,
        grid: true,
        ticks: 4,
        domain: diverging ? [-extent, extent] : [0, extent],
      },
      y: { label: null, domain: data.map((d) => d.label) },
      marks: [
        Plot.barX(data, {
          x: "value",
          y: "label",
          fill: diverging
            ? (d: { value: number }) =>
                d.value >= 0 ? (theme as ChartTheme).pair[0] : (theme as ChartTheme).pair[1]
            : (theme as ChartTheme).pair[0],
          // Rounded at the data end only, anchored to the baseline.
          rx: 3,
          insetTop: 2,
          insetBottom: 2,
          title: (d: { label: string; value: number; detail?: string }) =>
            `${d.label}\n${render(d.value)}${d.detail ? `\n${d.detail}` : ""}`,
        }),
        // Two marks rather than one, because `textAnchor` and `dx` are Plot
        // constants and not per-datum channels. The label has to sit past the
        // *end* of its bar, whichever side of zero that is: a single "start"
        // anchor puts every negative bar's label inside the bar, low-contrast
        // against the fill and clipped by the bar's own edge.
        Plot.text(
          data.filter((d) => d.value >= 0),
          {
            x: "value",
            y: "label",
            text: (d: { value: number }) => render(d.value),
            textAnchor: "start",
            dx: 6,
            fill: (theme as ChartTheme).muted,
            fontSize: 10,
          },
        ),
        Plot.text(
          data.filter((d) => d.value < 0),
          {
            x: "value",
            y: "label",
            text: (d: { value: number }) => render(d.value),
            textAnchor: "end",
            dx: -6,
            fill: (theme as ChartTheme).muted,
            fontSize: 10,
          },
        ),
        Plot.ruleX([0], { stroke: (theme as ChartTheme).grid }),
      ],
    };
  }, [data, xLabel, diverging, format, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}

/**
 * A cumulative curve -- what share of the dataset the *k* most common scaffolds
 * account for. A curve that reaches 1.0 in a handful of steps is a congeneric
 * series; one that climbs slowly is a diverse deck.
 */
export function CoverageCurveChart({
  coverage,
  height = 200,
  caption,
}: {
  coverage: number[];
  height?: number;
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const data = coverage.map((value, index) => ({ rank: index + 1, value }));
    return {
      ...baseOptions(theme as ChartTheme),
      x: { label: "scaffolds, most common first", grid: true, ticks: 5 },
      // `percent: true` would scale the values by 100 and leave them outside
      // this domain, which drew the whole curve clipped flat against the top of
      // the frame. The domain is the honest part -- a coverage curve read on
      // anything but a full 0-to-1 axis exaggerates whatever it shows -- so the
      // formatting moves to the ticks instead.
      y: {
        label: "share of compounds",
        domain: [0, 1],
        grid: true,
        ticks: 4,
        tickFormat: (value: number) => `${Math.round(value * 100)}%`,
      },
      marks: [
        Plot.areaY(data, {
          x: "rank",
          y: "value",
          fill: (theme as ChartTheme).pair[0],
          fillOpacity: 0.15,
        }),
        Plot.line(data, {
          x: "rank",
          y: "value",
          stroke: (theme as ChartTheme).pair[0],
          strokeWidth: 2,
          title: (d: { rank: number; value: number }) =>
            `top ${d.rank} scaffolds\n${(d.value * 100).toFixed(0)}% of compounds`,
        }),
      ],
    };
  }, [coverage, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}

export interface EpochLine {
  label: string;
  color: string;
  /** Secondary encoding beside color: SVG dash pattern, solid when omitted. */
  dash?: string;
  values: { epoch: number; value: number }[];
}

const EPOCH_MARGIN_RIGHT = 92;
const LABEL_GAP_PX = 13;

/** The y range a set of lines spans, padded so no line runs along the frame. */
function paddedExtent(lines: EpochLine[]): [number, number] {
  const values = lines.flatMap((line) => line.values.map((point) => point.value));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low) * 0.08 || Math.abs(high) * 0.05 || 0.05;
  return [low - pad, high + pad];
}

/**
 * Each line's end label, nudged apart so two lines ending close together do not print
 * their names on top of each other: a label's y in data units, after placing every
 * label at least LABEL_GAP_PX from the next in screen space.
 */
function endLabels(lines: EpochLine[], domain: [number, number], height: number) {
  const top = 22;
  const bottom = height - 34;
  const toPx = (value: number) =>
    bottom - ((value - domain[0]) / (domain[1] - domain[0])) * (bottom - top);
  const toValue = (px: number) =>
    domain[0] + ((bottom - px) / (bottom - top)) * (domain[1] - domain[0]);
  const ends = lines
    .filter((line) => line.values.length > 0)
    .map((line) => {
      const last = line.values[line.values.length - 1];
      return { label: line.label, epoch: last.epoch, px: toPx(last.value) };
    })
    .sort((a, b) => a.px - b.px);
  for (let index = 1; index < ends.length; index++) {
    ends[index].px = Math.max(ends[index].px, ends[index - 1].px + LABEL_GAP_PX);
  }
  return ends.map((end) => ({ label: end.label, epoch: end.epoch, value: toValue(end.px) }));
}

/**
 * Lines over a fit's epochs, on one axis: the losses on one chart, the validation
 * scores on another, never both on two scales. `kept` marks the epoch the fit keeps
 * (as the fit reports it) with a labelled rule; `epochs` fixes the x axis at the
 * fit's full length, so a live chart shows how far there is still to go.
 */
export function EpochCurveChart({
  lines,
  epochs,
  kept,
  yLabel,
  format = (value: number) => value.toFixed(3),
  height = 200,
  caption,
}: {
  lines: EpochLine[];
  epochs: number;
  kept?: number | null;
  yLabel: string;
  format?: (value: number) => string;
  height?: number;
  caption?: React.ReactNode;
}) {
  const theme = useChartTheme();
  const options = useCallback((): Plot.PlotOptions => {
    const t = theme as ChartTheme;
    const domain = paddedExtent(lines);
    const byEpoch = new Map<number, Record<string, number>>();
    for (const line of lines) {
      for (const point of line.values) {
        byEpoch.set(point.epoch, { ...byEpoch.get(point.epoch), [line.label]: point.value });
      }
    }
    // One row per epoch, so the tooltip lists every series at the epoch under the pointer.
    const rows = [...byEpoch.entries()]
      .sort(([a], [b]) => a - b)
      .map(([epoch, values]) => ({ epoch, values }));
    return {
      ...baseOptions(t),
      marginRight: EPOCH_MARGIN_RIGHT,
      x: {
        label: "epoch",
        domain: [1, Math.max(epochs, 2)],
        ticks: Math.min(6, Math.max(epochs, 2) - 1),
        tickFormat: "d",
      },
      y: { label: yLabel, domain, grid: true, ticks: 4 },
      marks: [
        ...(kept != null
          ? [
              Plot.ruleX([kept], { stroke: t.muted, strokeDasharray: "2,3" }),
              Plot.text([kept], {
                x: (epoch: number) => epoch,
                frameAnchor: "top",
                dy: -10,
                text: () => "kept",
                fill: t.muted,
              }),
            ]
          : []),
        ...lines.map((line) =>
          Plot.line(line.values, {
            x: "epoch",
            y: "value",
            stroke: line.color,
            strokeWidth: 2,
            strokeDasharray: line.dash,
          }),
        ),
        Plot.text(endLabels(lines, domain, height), {
          x: "epoch",
          y: "value",
          text: "label",
          fill: t.text,
          textAnchor: "start",
          dx: 6,
        }),
        Plot.ruleX(rows, Plot.pointerX({ x: "epoch", stroke: t.grid })),
        Plot.tip(
          rows,
          Plot.pointerX({
            x: "epoch",
            frameAnchor: "top-left",
            title: (row: { epoch: number; values: Record<string, number> }) =>
              [
                `epoch ${row.epoch}`,
                ...lines.map((line) =>
                  line.label in row.values
                    ? `${format(row.values[line.label])}  ${line.label}`
                    : `N/A  ${line.label}`,
                ),
              ].join("\n"),
          }),
        ),
      ],
    };
  }, [lines, epochs, kept, yLabel, format, height, theme]);

  if (!theme) return <Pending height={height} />;
  return <PlotFigure options={options} height={height} caption={caption} />;
}
