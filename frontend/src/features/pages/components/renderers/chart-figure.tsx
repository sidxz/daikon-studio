"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  type ReferenceLineSegment,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { format } from "date-fns";

import {
  MAX_SERIES,
  type TableRow,
  isDateColumn,
  isNumericColumn,
  parseDateCell,
  parseTable,
  resolveColumns,
} from "@/features/pages/lib/parse-table";

export type ChartKind = "bar" | "line" | "scatter" | "radar" | "pie";

export type RefLine =
  | { axis: "x" | "y"; value: number; label?: string }
  // Named rather than a coordinate pair: "diagonal" is ROC's (0,0)–(1,1) chance
  // line, "identity" is a parity plot's y=x drawn across the data's own range.
  | { segment: "diagonal" | "identity" };

export type ChartOptions = {
  logX?: boolean;
  logY?: boolean;
  errorColumn?: string | null;
  horizontal?: boolean;
  step?: boolean;
  stacked?: boolean;
  refLines?: RefLine[];
  /** Scatter only: split the points into one group per distinct value of this
   *  column, each with its own colour AND marker shape. This is how a category
   *  like "Catalyst" or "Series" becomes readable — as groups of points rather
   *  than a numeric series, which is what the `series` list models. */
  groupBy?: string | null;
  /** Pin an axis to a fixed range instead of fitting it to the data. Either end
   *  can be set alone. Pinning is what lets two charts be compared directly —
   *  auto-fitted axes silently rescale per chart and make unlike things look
   *  alike. The x pair applies to scatter only; bar and line have a categorical
   *  x axis, which has no numeric range to pin. */
  xMin?: number | null;
  xMax?: number | null;
  yMin?: number | null;
  yMax?: number | null;
  /** Axis titles. Undefined means "derive from the data" (see axisTitles); an
   *  empty string means the author deliberately cleared it. */
  xLabel?: string | null;
  yLabel?: string | null;
};

/** Fixed slots from @structflo/daikon-design-tokens (light + dark). Indexed by
 *  series position and never cycled — the 6th series is prevented upstream by
 *  MAX_SERIES rather than by wrapping around to chart-1. */
export const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

/**
 * Marker shapes, paired 1:1 with SERIES_COLORS. Shape is a *redundant* encoding
 * of the same grouping the colour carries, so the chart still reads when the
 * colours don't — colour-vision deficiency, greyscale print, a projector.
 */
export const SERIES_SHAPES = ["circle", "triangle", "square", "diamond", "cross"] as const;

/** Part-to-whole reads at a glance only up to ~6 wedges; past that the chart is
 *  a table. The 6th slot is always the folded remainder. */
export const PIE_MAX_SLICES = 6;

const AXIS_PROPS = {
  stroke: "var(--border)",
  tick: { fill: "var(--muted-foreground)", fontSize: 12 },
} as const;

const TOOLTIP_STYLE = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "0.375rem",
  color: "var(--popover-foreground)",
  fontSize: 12,
};

/**
 * What the axes are called. A chart with unlabelled axes is not a self-contained
 * figure — a reader six months later has only the caption to go on — so these
 * default to the column names that are already in the pasted data rather than
 * waiting for someone to type them.
 *
 * The y title is left blank for a multi-series chart: the legend already names
 * each series, and a single title over several measures would be a lie. `??`
 * rather than `||` so an author can deliberately clear a title.
 */
export function axisTitles(
  options: Pick<ChartOptions, "xLabel" | "yLabel">,
  xKey: string,
  keys: string[],
): { x: string; y: string } {
  return {
    x: options.xLabel ?? xKey,
    y: options.yLabel ?? (keys.length === 1 ? keys[0] : ""),
  };
}

/** Tick granularity follows the span: a two-year series wants months, a two-day
 *  one wants hours. Without this every tick reads "15 Jan 2026 03:27". */
export function dateTickFormat(spanMs: number): string {
  const day = 86_400_000;
  if (spanMs > 3 * 365 * day) return "yyyy";
  if (spanMs > 90 * day) return "MMM yyyy";
  if (spanMs > 2 * day) return "d MMM";
  return "HH:mm";
}

/**
 * A ScatterChart reads x and y from its *axes*, not from the Scatter — giving
 * <Scatter> a dataKey makes Recharts read it as the point-size channel, which
 * silently shrinks every marker to a speck. So each series is remapped onto the
 * common {x, y} shape the axes point at.
 */
export function toScatterPoints(
  rows: TableRow[],
  xKey: string,
  seriesKey: string,
  errorColumn?: string | null,
) {
  return rows.map((r) => ({
    x: r[xKey],
    y: r[seriesKey],
    e: errorColumn ? r[errorColumn] : undefined,
  }));
}

/**
 * Split rows into one group per distinct value of `column`, in order of first
 * appearance so the legend order matches the data rather than being alphabetised.
 *
 * ponytail: capped at MAX_SERIES groups because there are only five hue/shape
 * slots and a 6th would have to reuse one. The overflow is NOT silently hidden
 * — the dialog warns when it truncates, and every row stays visible in the
 * chart's "Show data" table. Raise the cap only by adding real slots.
 */
export function groupRowsBy(rows: TableRow[], column: string): { key: string; rows: TableRow[] }[] {
  const groups = new Map<string, TableRow[]>();
  for (const row of rows) {
    const key = String(row[column] ?? "");
    const bucket = groups.get(key);
    if (bucket) bucket.push(row);
    else groups.set(key, [row]);
  }
  return [...groups].slice(0, MAX_SERIES).map(([key, groupRows]) => ({ key, rows: groupRows }));
}

/** The span y=x is drawn across, over every plotted value so the line reaches
 *  both corners of the data rather than only one axis's range. */
export function dataExtent(rows: TableRow[], xKey: string, keys: string[]): [number, number] {
  const nums = rows
    .flatMap((r) => [r[xKey], ...keys.map((k) => r[k])])
    .filter((v): v is number => typeof v === "number");
  return nums.length > 0 ? [Math.min(...nums), Math.max(...nums)] : [0, 1];
}

/** Keeps the largest slices and sums everything else into one muted "Other". */
export function foldPieSlices(rows: TableRow[], nameKey: string, valueKey: string): TableRow[] {
  if (rows.length <= PIE_MAX_SLICES) return rows;
  const num = (r: TableRow) => (typeof r[valueKey] === "number" ? (r[valueKey] as number) : 0);
  const sorted = [...rows].sort((a, b) => num(b) - num(a));
  const kept = sorted.slice(0, PIE_MAX_SLICES - 1);
  const rest = sorted.slice(PIE_MAX_SLICES - 1).reduce((sum, r) => sum + num(r), 0);
  return [...kept, { [nameKey]: "Other", [valueKey]: rest } as TableRow];
}

/**
 * Log scales reject values <= 0, and Recharts needs an explicit numeric type.
 *
 * The domain is the data's own range, NOT ["auto", "auto"]: on a log scale
 * "auto" snaps to a round power, so a series topping out at 1400 got an axis
 * ending at 1000 — and because allowDataOverflow was also set, the 1400 point
 * was clipped away rather than extending the axis. Overflow is left off here so
 * nothing is silently dropped; a deliberate pin re-enables it via axisRange.
 */
function axisScale(log: boolean | undefined) {
  return log ? ({ scale: "log", domain: ["dataMin", "dataMax"], type: "number" } as const) : {};
}

/**
 * Scatter axes fit the data instead of anchoring at zero. Recharts' numeric
 * default is [0, dataMax], which for values far from the origin (reaction
 * temperatures around 150-160 °C, say) squeezes every point into one corner and
 * wastes the plot. A scatter compares positions, so a non-zero baseline is
 * honest here — unlike a bar chart, where the bar length IS the value and a
 * truncated axis misstates it. Bars therefore keep the zero baseline.
 */
const SCATTER_DOMAIN = { domain: ["dataMin", "dataMax"] } as const;

/**
 * Axis range props. Either bound may be pinned on its own; the other keeps its
 * automatic behaviour. `allowDataOverflow` is the switch that makes a pin mean
 * something: without it Recharts widens the domain to fit stray points, so
 * "lock x to 0-100" would quietly become 0-120 the moment a point exceeded it.
 * With it, out-of-range points are clipped and the axis says what it says.
 */
// Recharts types `domain` as a fixed 2-tuple, not an array, so this has to be
// annotated — an inferred (string | number)[] is rejected.
type DomainPair = readonly [number | string, number | string];

export function axisRange(
  min: number | null | undefined,
  max: number | null | undefined,
  fallback: DomainPair,
): { domain: DomainPair; allowDataOverflow?: boolean } {
  const pinned = min != null || max != null;
  if (!pinned) return { domain: fallback };
  return { domain: [min ?? fallback[0], max ?? fallback[1]] as const, allowDataOverflow: true };
}

// Fitting the domain exactly to the data puts the extreme points ON the axis
// lines, where the markers get clipped in half. A few pixels of inset keeps
// them whole without widening the domain (which would misstate the range).
/** Bar and line keep Recharts' zero-anchored default when nothing is pinned. */
const BASELINE_DOMAIN = [0, "auto"] as const;

/**
 * It is dropped when that axis is pinned, because then the inset is drawable
 * space *outside* the stated range: a point just past the bound rendered a
 * sliver of a marker beyond the axis end instead of disappearing.
 */
function scatterPadding(axis: "x" | "y", pinned: boolean) {
  if (pinned) return {};
  return axis === "x"
    ? ({ padding: { left: 14, right: 14 } } as const)
    : ({ padding: { top: 14, bottom: 14 } } as const);
}

/**
 * A line chart's x axis is categorical by default, which spaces points evenly by
 * row instead of by value — correct for month names, wrong for an ROC's false
 * positive rate, where 0 -> 0.02 would get the same width as 0.6 -> 1 and bend
 * the curve. A numeric x column therefore gets a real numeric axis, which also
 * makes an x range pinnable.
 */
function lineXAxisProps(numeric: boolean, options: ChartOptions) {
  if (!numeric) return {};
  return {
    type: "number" as const,
    ...axisRange(options.xMin, options.xMax, ["dataMin", "dataMax"] as const),
  };
}

/**
 * `segment` takes literal coordinates only — unlike an axis `domain`, Recharts
 * does not resolve "dataMin"/"dataMax" here, and passing them draws a line
 * across nonsense positions. So y=x is computed from the plotted values.
 *
 * The return is annotated because ReferenceLine's value generics default to
 * `any` and would otherwise narrow to the literal of whatever appears first,
 * rejecting the coordinate at the other end.
 */
export function segmentFor(
  segment: "diagonal" | "identity",
  extent: [number, number],
): ReferenceLineSegment {
  // "diagonal" is ROC/PR's chance line, which is the unit square by definition
  // regardless of where the plotted points happen to fall.
  const [lo, hi] = segment === "diagonal" ? [0, 1] : extent;
  return [
    { x: lo, y: lo },
    { x: hi, y: hi },
  ];
}

function renderRefLines(
  refLines: RefLine[] | undefined,
  extent: [number, number],
  // A pinned axis must win over a reference line: growing the domain to fit the
  // line would undo the range the author explicitly asked for, so an
  // out-of-range line is clipped instead of dropped or honoured.
  axisPinned: boolean,
) {
  const ifOverflow = axisPinned ? "hidden" : "extendDomain";
  return (refLines ?? []).map((r, i) =>
    "segment" in r ? (
      <ReferenceLine
        key={`ref-${i}`}
        segment={segmentFor(r.segment, extent)}
        stroke="var(--muted-foreground)"
        strokeDasharray="4 4"
        // Without this a line whose endpoints fall outside the auto domain is
        // silently dropped whole — a y=x across data the axes don't already
        // span, or a p=0.05 threshold every point clears. Growing the axes to
        // include it is what the reader expects in both cases.
        ifOverflow={ifOverflow}
      />
    ) : (
      <ReferenceLine
        key={`ref-${i}`}
        {...(r.axis === "x" ? { x: r.value } : { y: r.value })}
        stroke="var(--muted-foreground)"
        strokeDasharray="4 4"
        // Without this a line whose endpoints fall outside the auto domain is
        // silently dropped whole — a y=x across data the axes don't already
        // span, or a p=0.05 threshold every point clears. Growing the axes to
        // include it is what the reader expects in both cases.
        ifOverflow={ifOverflow}
        label={
          r.label
            ? {
                value: r.label,
                fill: "var(--muted-foreground)",
                fontSize: 11,
                // Above the line at its start, not centred on it: a threshold is
                // usually drawn exactly where the data crosses it, so a centred
                // label lands on top of the very series it describes.
                position: "insideTopLeft",
              }
            : undefined
        }
      />
    ),
  );
}

/**
 * Renders one embedded chart. Everything it needs comes from the node's attrs;
 * it holds no state and fetches nothing, so the editor and the read-only page
 * view render identically.
 *
 * A legend appears from two series up — with one series the caption already
 * names it, and a one-entry legend is chrome.
 */
export function ChartFigure({
  kind,
  raw,
  x,
  series,
  options,
}: {
  kind: ChartKind;
  raw: string;
  x: string | null;
  series: string[] | null;
  options: ChartOptions;
}) {
  const { table, roles } = useMemo(() => {
    const t = parseTable(raw);
    return { table: t, roles: resolveColumns(t, x, series) };
  }, [raw, x, series]);

  const { rows } = table;
  const { x: xKey, series: keys } = roles;

  // Bars stay categorical even for dates — a bar IS a category, and evenly
  // spaced bars over dates are not misleading. Line and scatter are where an
  // unrecognised date silently became a row index.
  const xIsDate = useMemo(
    () => (kind === "line" || kind === "scatter") && isDateColumn(table, xKey),
    [kind, table, xKey],
  );

  // Rows with the date column turned into epoch milliseconds, which is what a
  // time scale plots. Rows whose x will not parse are dropped rather than
  // pinned to 1970; isDateColumn guarantees that is only ever a blank cell.
  const plotRows = useMemo(() => {
    if (!xIsDate) return rows;
    return rows.flatMap((r) => {
      const t = parseDateCell(r[xKey]);
      return t === null ? [] : [{ ...r, [xKey]: t }];
    });
  }, [rows, xKey, xIsDate]);

  const slices = useMemo(
    () => (kind === "pie" && keys.length > 0 ? foldPieSlices(rows, xKey, keys[0]) : []),
    [kind, rows, xKey, keys],
  );

  // Grouping wins over multi-series when both could apply: a "Color by" choice
  // is explicit, and one measure split across categories is what the author
  // asked for. Each entry becomes its own <Scatter>, hence its own colour,
  // shape and legend row.
  const scatterSeries = useMemo(() => {
    if (kind !== "scatter" || keys.length === 0) return [];
    const groupBy = options.groupBy;
    if (groupBy) {
      return groupRowsBy(plotRows, groupBy).map((g) => ({
        key: g.key,
        points: toScatterPoints(g.rows, xKey, keys[0], options.errorColumn),
      }));
    }
    return keys.map((k) => ({
      key: k,
      points: toScatterPoints(plotRows, xKey, k, options.errorColumn),
    }));
  }, [kind, plotRows, xKey, keys, options.errorColumn, options.groupBy]);

  const extent = useMemo(() => dataExtent(rows, xKey, keys), [rows, xKey, keys]);
  const xIsNumeric = useMemo(() => isNumericColumn(table, xKey), [table, xKey]);
  if (rows.length === 0 || keys.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground">
        No chart data yet.
      </div>
    );
  }

  const axisPinned =
    options.xMin != null || options.xMax != null || options.yMin != null || options.yMax != null;

  // Radar and pie have no cartesian axes, so an axis title on them is nonsense —
  // a pie was captioned "outcome" along the bottom and "compounds" up the side.
  const cartesian = kind === "bar" || kind === "line" || kind === "scatter";
  const titles = cartesian ? axisTitles(options, xKey, keys) : { x: "", y: "" };

  // A time scale needs epoch numbers plus a formatter, or every tick reads as a
  // 13-digit millisecond count.
  const dateSpan = xIsDate
    ? (() => {
        const ts = plotRows.map((r) => r[xKey] as number);
        return Math.max(...ts) - Math.min(...ts);
      })()
    : 0;
  const dateFmt = dateTickFormat(dateSpan);
  const dateAxis = xIsDate
    ? {
        type: "number" as const,
        scale: "time" as const,
        domain: ["dataMin", "dataMax"] as const,
        tickFormatter: (v: number) => format(v, dateFmt),
      }
    : {};
  // The axis formatter does not reach the tooltip, so the date is restated here
  // in full — a tick may read "Feb" while the point is the 3rd.
  const dateTooltip = xIsDate
    ? {
        labelFormatter: (v: unknown) =>
          typeof v === "number" ? format(v, "d MMM yyyy") : String(v),
        formatter: (value: unknown, name: unknown) =>
          name === titles.x && typeof value === "number"
            ? [format(value, "d MMM yyyy"), name as string]
            : [value as number, name as string],
      }
    : {};

  // Two or more *plotted things* — which for a grouped scatter is the groups,
  // not the single measure they all share.
  const showLegend = (kind === "scatter" ? scatterSeries.length : keys.length) > 1;
  const err = options.errorColumn ?? undefined;

  // An array, deliberately NOT a fragment: Recharts locates axes, legend and
  // reference lines by walking its children, and React.Children flattens arrays
  // into siblings but leaves a fragment as one opaque child.
  const common = [
    <CartesianGrid key="grid" strokeDasharray="3 3" stroke="var(--border)" />,
    <Tooltip
      key="tip"
      contentStyle={TOOLTIP_STYLE}
      cursor={{ stroke: "var(--border)" }}
      {...dateTooltip}
    />,
    ...(showLegend
      ? [<Legend key="legend" wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }} />]
      : []),
    ...renderRefLines(options.refLines, extent, axisPinned),
  ];

  return (
    // Axis titles are HTML around the chart rather than Recharts' own `label`:
    // an "insideBottom" label is drawn past the axis band and overlaps the
    // legend by ~7px no matter how the axis height or chart margin is set. As
    // plain text they also stay selectable, translatable and readable to a
    // screen reader, which SVG labels are not.
    <div className="flex items-stretch gap-1">
      {titles.y ? (
        <div className="flex items-center">
          <span className="rotate-180 text-xs text-muted-foreground [writing-mode:vertical-rl]">
            {titles.y}
          </span>
        </div>
      ) : null}
      <div className="min-w-0 flex-1">
        {/* aspect rather than a fixed height: the figure's width comes from the
            embed's size preset, and the chart keeps its proportions at any of them. */}
        <ResponsiveContainer width="100%" aspect={16 / 9}>
          {kind === "bar" ? (
            <BarChart
              data={rows}
              layout={options.horizontal ? "vertical" : "horizontal"}
              barGap={2}
            >
              {common}
              {options.horizontal ? (
                // No axisScale here: a bar is drawn from the zero baseline to its
                // value, and log(0) is undefined, so a log scale leaves Recharts
                // with no baseline and it renders no bars at all. The dialog does
                // not offer the option; this ignores it on any chart that stored
                // one before, rather than rendering an empty plot.
                <XAxis
                  type="number"
                  {...AXIS_PROPS}
                  {...axisRange(options.xMin, options.xMax, BASELINE_DOMAIN)}
                />
              ) : (
                <XAxis dataKey={xKey} {...AXIS_PROPS} />
              )}
              {options.horizontal ? (
                <YAxis type="category" dataKey={xKey} {...AXIS_PROPS} />
              ) : (
                // See the horizontal branch: bars and log scales are incompatible.
                <YAxis
                  {...AXIS_PROPS}
                  {...axisRange(options.yMin, options.yMax, BASELINE_DOMAIN)}
                />
              )}
              {keys.map((k, i) => (
                <Bar
                  key={k}
                  dataKey={k}
                  fill={SERIES_COLORS[i]}
                  stackId={options.stacked ? "stack" : undefined}
                  radius={options.horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}
                  isAnimationActive={false}
                >
                  {err ? (
                    <ErrorBar dataKey={err} stroke="var(--foreground)" strokeWidth={2} width={4} />
                  ) : null}
                </Bar>
              ))}
            </BarChart>
          ) : kind === "line" ? (
            <LineChart data={plotRows}>
              {common}
              <XAxis
                dataKey={xKey}
                name={titles.x}
                {...AXIS_PROPS}
                {...axisScale(options.logX)}
                {...lineXAxisProps(xIsNumeric, options)}
                {...dateAxis}
              />
              <YAxis
                name={titles.y}
                {...AXIS_PROPS}
                {...axisScale(options.logY)}
                {...axisRange(options.yMin, options.yMax, BASELINE_DOMAIN)}
              />
              {keys.map((k, i) => (
                <Line
                  key={k}
                  type={options.step ? "stepAfter" : "monotone"}
                  dataKey={k}
                  stroke={SERIES_COLORS[i]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                >
                  {err ? (
                    <ErrorBar dataKey={err} stroke={SERIES_COLORS[i]} strokeWidth={2} width={4} />
                  ) : null}
                </Line>
              ))}
            </LineChart>
          ) : kind === "scatter" ? (
            <ScatterChart>
              {common}
              {/* Scatter needs a numeric x — a categorical column would collapse
              every point onto one tick, which is what the bar chart is for. */}
              <XAxis
                type="number"
                dataKey="x"
                name={titles.x}
                {...AXIS_PROPS}
                {...SCATTER_DOMAIN}
                {...scatterPadding("x", options.xMin != null || options.xMax != null)}
                {...axisScale(options.logX)}
                {...axisRange(options.xMin, options.xMax, SCATTER_DOMAIN.domain)}
                {...dateAxis}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={titles.y || undefined}
                {...AXIS_PROPS}
                {...SCATTER_DOMAIN}
                {...scatterPadding("y", options.yMin != null || options.yMax != null)}
                {...axisScale(options.logY)}
                {...axisRange(options.yMin, options.yMax, SCATTER_DOMAIN.domain)}
              />
              {scatterSeries.map((s, i) => (
                <Scatter
                  key={s.key}
                  name={s.key}
                  data={s.points}
                  fill={SERIES_COLORS[i]}
                  shape={SERIES_SHAPES[i]}
                  // Without this the legend draws a circle for every group, so the
                  // marker shape — the whole point of the redundant encoding — is
                  // not decodable from the legend. Every SERIES_SHAPES value is
                  // also a valid legendType.
                  legendType={SERIES_SHAPES[i]}
                  isAnimationActive={false}
                >
                  {err ? (
                    <ErrorBar dataKey="e" stroke={SERIES_COLORS[i]} strokeWidth={2} width={4} />
                  ) : null}
                </Scatter>
              ))}
            </ScatterChart>
          ) : kind === "radar" ? (
            <RadarChart data={rows}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis
                dataKey={xKey}
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              />
              <PolarRadiusAxis tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              {showLegend ? (
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }} />
              ) : null}
              {keys.map((k, i) => (
                <Radar
                  key={k}
                  name={k}
                  dataKey={k}
                  stroke={SERIES_COLORS[i]}
                  strokeWidth={2}
                  fill={SERIES_COLORS[i]}
                  fillOpacity={0.2}
                  isAnimationActive={false}
                />
              ))}
            </RadarChart>
          ) : (
            <PieChart>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }} />
              {/* Pie plots one measure; extra selected series are ignored rather
              than silently summed. Folded once and reused so the wedges and
              their Cells can never fall out of step. */}
              <Pie
                data={slices}
                dataKey={keys[0]}
                nameKey={xKey}
                innerRadius="45%"
                outerRadius="75%"
                // 2px of surface between wedges, per the mark spec.
                paddingAngle={1}
                stroke="var(--background)"
                strokeWidth={2}
                isAnimationActive={false}
              >
                {slices.map((_, i) => (
                  // Colored by position, never cycled: past the five hue slots a
                  // wedge goes muted rather than repeating chart-1. With folding on
                  // that muted wedge is always the "Other" remainder.
                  <Cell key={i} fill={SERIES_COLORS[i] ?? "var(--muted-foreground)"} />
                ))}
              </Pie>
            </PieChart>
          )}
        </ResponsiveContainer>
        {titles.x ? (
          <p className="mt-0.5 text-center text-xs text-muted-foreground">{titles.x}</p>
        ) : null}
      </div>
    </div>
  );
}
