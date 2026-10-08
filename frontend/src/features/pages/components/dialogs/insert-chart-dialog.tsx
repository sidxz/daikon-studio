"use client";

import type { Editor } from "@tiptap/react";
import { Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import {
  MAX_SERIES,
  isDateColumn,
  isNumericColumn,
  parseTable,
  resolveColumns,
} from "@/features/pages/lib/parse-table";
import { Button } from "@/shared/components/ui/button";
import { Checkbox } from "@/shared/components/ui/checkbox";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/shared/components/ui/field";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Textarea } from "@/shared/components/ui/textarea";

import { CHART_PRESETS } from "../chart-presets";
import type { EmbedEditRequest } from "../embed-toolbar";
import {
  ChartFigure,
  type ChartKind,
  type ChartOptions,
  type RefLine,
  axisTitles,
} from "../renderers/chart-figure";
import { EmbedDialogFrame } from "./embed-dialog-frame";

const KINDS: { value: ChartKind; label: string }[] = [
  { value: "bar", label: "Bar" },
  { value: "line", label: "Line" },
  { value: "scatter", label: "Scatter" },
  { value: "radar", label: "Radar" },
  { value: "pie", label: "Pie" },
];

// Fixed height, matching the data textarea beside it: neither typing, switching
// chart type, nor a parse error may resize the dialog (no layout jump).
const PREVIEW_CLASS =
  "flex h-44 shrink-0 items-center justify-center overflow-hidden rounded-md border border-border bg-muted/30 p-2 text-sm text-muted-foreground";

const DEFAULT_SAMPLE = "compound,ic50_nm\nA,120\nB,45\nC,310";

const SEGMENTS = [
  { value: "diagonal", label: "Chance diagonal (0,0)-(1,1)" },
  { value: "identity", label: "Identity y = x" },
] as const satisfies readonly { value: "diagonal" | "identity"; label: string }[];

const AXIS_BOUNDS = [
  { key: "xMin", axis: "x", label: "X min" },
  { key: "xMax", axis: "x", label: "X max" },
  { key: "yMin", axis: "y", label: "Y min" },
  { key: "yMax", axis: "y", label: "Y max" },
] as const satisfies readonly {
  key: "xMin" | "xMax" | "yMin" | "yMax";
  axis: "x" | "y";
  label: string;
}[];

/**
 * Insert-a-chart dialog. `preset` seeds the chart type, options and placeholder
 * from CHART_PRESETS when the author picked a named chart from the slash menu;
 * `editing` seeds from an existing node instead. Both are read once at mount —
 * EmbedDialogFrame remounts this component on every open, so no sync effect.
 */
export function InsertChartDialog({
  editor,
  open,
  onOpenChange,
  editing,
  preset,
}: {
  editor: Editor;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editing?: EmbedEditRequest;
  preset?: string;
}) {
  const seed = preset ? CHART_PRESETS[preset] : undefined;

  const [kind, setKind] = useState<ChartKind>(
    (editing?.attrs.kind as ChartKind) ?? seed?.kind ?? "bar",
  );
  const [raw, setRaw] = useState((editing?.attrs.raw as string) ?? "");
  const [x, setX] = useState<string | null>((editing?.attrs.x as string | null) ?? null);
  const [series, setSeries] = useState<string[] | null>(
    (editing?.attrs.series as string[] | null) ?? null,
  );
  const [options, setOptions] = useState<ChartOptions>(
    (editing?.attrs.options as ChartOptions) ?? seed?.options ?? {},
  );
  const [caption, setCaption] = useState((editing?.attrs.caption as string) ?? "");

  const table = useMemo(() => parseTable(raw), [raw]);
  const roles = useMemo(() => resolveColumns(table, x, series), [table, x, series]);
  const numericColumns = useMemo(
    () => table.columns.filter((c) => isNumericColumn(table, c)),
    [table],
  );
  const plottableColumns = numericColumns.filter((c) => c !== roles.x);
  // Anything non-numeric is a label, which is exactly what you'd group by.
  const categoryColumns = table.columns.filter((c) => !isNumericColumn(table, c));
  // A bar's x axis is categorical, so there is no numeric range to pin — unless
  // the bars are horizontal, which swaps which axis carries the measure. A line's
  // x axis is numeric only when the column is (see lineXAxisProps). Radar and pie
  // have no cartesian axes at all.
  const xRangeApplies =
    kind === "scatter" ||
    (kind === "bar" && !!options.horizontal) ||
    (kind === "line" && !!roles.x && isNumericColumn(table, roles.x));
  const yRangeApplies =
    kind === "scatter" || ((kind === "bar" || kind === "line") && !options.horizontal);
  // A date column is a label, but it belongs on the time axis rather than in
  // "Color by" — grouping by every distinct timestamp is never what you want.
  const dateColumns = table.columns.filter((c) => isDateColumn(table, c));
  const groupableColumns = categoryColumns.filter((c) => !dateColumns.includes(c));
  const xIsDate = !!roles.x && dateColumns.includes(roles.x);
  const titles = axisTitles(options, roles.x, roles.series);

  // Reference lines are only meaningful on a numeric axis, which is exactly what
  // the range pins already work out. The named segments need BOTH axes numeric.
  const refLinesApply = xRangeApplies || yRangeApplies;
  const segmentsApply = xRangeApplies && yRangeApplies;
  const refLines = options.refLines ?? [];
  const segments = refLines.filter(
    (r): r is Extract<RefLine, { segment: string }> => "segment" in r,
  );
  const valueLines = refLines.filter(
    (r): r is Extract<RefLine, { axis: "x" | "y" }> => !("segment" in r),
  );
  // Rebuilt whole rather than spliced in place: the stored array mixes segments
  // and value lines, so a display index is not a storage index.
  const writeRefLines = (segs: typeof segments, vals: typeof valueLines) =>
    setOption("refLines", [...segs, ...vals]);

  const groupCount = options.groupBy
    ? new Set(table.rows.map((r) => String(r[options.groupBy!] ?? ""))).size
    : 0;

  const hasData = table.rows.length > 0 && roles.series.length > 0;
  const setOption = <K extends keyof ChartOptions>(key: K, value: ChartOptions[K]) =>
    setOptions((o) => ({ ...o, [key]: value }));

  // Toggling a column edits the *resolved* list, so the first click on a
  // never-touched chart starts from what is actually plotted rather than from null.
  function toggleSeries(column: string) {
    const current = roles.series;
    const next = current.includes(column)
      ? current.filter((c) => c !== column)
      : [...current, column].slice(0, MAX_SERIES);
    setSeries(next);
  }

  function confirm() {
    if (!hasData || editor.isDestroyed) return;
    const attrs = {
      kind,
      raw,
      x: roles.x || null,
      series: roles.series,
      options,
      caption: caption.trim() || null,
      width: (editing?.attrs.width as string) ?? "full",
    };
    if (editing) {
      editing.onUpdate(attrs);
    } else {
      editor.chain().focus().insertContent({ type: "chart", attrs }).run();
    }
    onOpenChange(false);
  }

  return (
    <EmbedDialogFrame
      open={open}
      onClose={() => onOpenChange(false)}
      title={editing ? "Edit chart" : seed ? `Insert ${seed.label.toLowerCase()}` : "Insert chart"}
      description={seed ? seed.hint : "Paste a small table — the first row is the column headers."}
      confirmLabel={editing ? "Save changes" : "Insert chart"}
      canConfirm={hasData}
      onConfirm={confirm}
      className="sm:max-w-2xl"
    >
      <div className="flex max-h-[70vh] flex-col gap-4 overflow-y-auto">
        <FieldSet>
          <FieldLegend variant="label">Chart type</FieldLegend>
          <div className="flex flex-wrap gap-1">
            {KINDS.map((k) => (
              <Button
                key={k.value}
                type="button"
                size="sm"
                variant={kind === k.value ? "default" : "outline"}
                aria-pressed={kind === k.value}
                onClick={() => setKind(k.value)}
              >
                {k.label}
              </Button>
            ))}
          </div>
        </FieldSet>

        {/* Data and preview side by side, at matching heights: the preview is
            the dialog's most useful feedback, and stacked below the options it
            fell off the bottom of the scroll area on a laptop screen — you had
            to scroll the dialog to see what you were building. */}
        <div className="grid gap-4 sm:grid-cols-2">
          <Field>
            <FieldLabel htmlFor="insert-chart-data">Data (CSV or tab-separated)</FieldLabel>
            <Textarea
              id="insert-chart-data"
              className="h-44 resize-none font-mono text-xs"
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
              placeholder={seed?.sample ?? DEFAULT_SAMPLE}
            />
          </Field>

          <FieldSet>
            <FieldLegend variant="label">Preview</FieldLegend>
            <div className={PREVIEW_CLASS}>
              {!raw.trim() ? (
                <p className="px-2 text-center">Paste a table to preview the chart</p>
              ) : !hasData ? (
                <p className="px-2 text-center">
                  No numeric column to plot yet — check the header row and the data.
                </p>
              ) : (
                // The chart keeps its own 16:9 aspect, so it needs a width to
                // measure against inside the fixed-height preview box.
                <div className="w-full">
                  <ChartFigure
                    kind={kind}
                    raw={raw}
                    x={roles.x}
                    series={roles.series}
                    options={options}
                  />
                </div>
              )}
            </div>
          </FieldSet>
        </div>

        {/* Rendered always, disabled until the data parses — appearing on first
            valid paste would resize the dialog. */}
        <div className="grid grid-cols-2 gap-4">
          <Field>
            <FieldLabel htmlFor="insert-chart-x">X axis column</FieldLabel>
            <Select
              value={roles.x || undefined}
              onValueChange={setX}
              disabled={table.columns.length === 0}
            >
              <SelectTrigger id="insert-chart-x">
                <SelectValue placeholder="Paste data first" />
              </SelectTrigger>
              <SelectContent>
                {table.columns.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          {/* Only numeric columns: a text column can't be a series, and offering
              it as one was a footgun — you'd tick "Catalyst" and plot nothing.
              Text columns belong in "Color by" below. */}
          <FieldSet>
            <FieldLegend variant="label">{`Series (up to ${MAX_SERIES})`}</FieldLegend>
            <div className="flex h-9 flex-wrap items-center gap-3 overflow-x-auto">
              {table.columns.length === 0 ? (
                <span className="text-sm text-muted-foreground">Paste data first</span>
              ) : plottableColumns.length === 0 ? (
                <span className="text-sm text-muted-foreground">No numeric column</span>
              ) : (
                plottableColumns.map((c) => (
                  <label key={c} className="flex items-center gap-1.5 text-sm">
                    <Checkbox
                      checked={roles.series.includes(c)}
                      onCheckedChange={() => toggleSeries(c)}
                    />
                    {c}
                  </label>
                ))
              )}
            </div>
          </FieldSet>
        </div>

        {/* Only the switches that mean something for the chosen mark. */}
        <FieldSet>
          <FieldLegend variant="label">Options</FieldLegend>
          <div className="flex flex-wrap items-center gap-4">
            {kind === "scatter" ? (
              <div className="flex items-center gap-1.5">
                <Label htmlFor="insert-chart-groupby" className="text-sm font-normal">
                  Color by
                </Label>
                <Select
                  value={options.groupBy ?? "none"}
                  onValueChange={(v) => setOption("groupBy", v === "none" ? null : v)}
                  disabled={groupableColumns.length === 0}
                >
                  <SelectTrigger id="insert-chart-groupby" className="h-8 w-36">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    {groupableColumns.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ) : null}
            {(kind === "scatter" || kind === "line") && !xIsDate ? (
              <label className="flex items-center gap-1.5 text-sm">
                <Checkbox
                  checked={!!options.logX}
                  onCheckedChange={(v) => setOption("logX", v === true)}
                />
                Log X
              </label>
            ) : null}
            {/* Not offered for bars: a bar runs from the zero baseline to its
                value and log(0) is undefined, so a log scale renders no bars at
                all. Plot a ratio, or use a scatter, when the values span decades. */}
            {kind === "line" || kind === "scatter" ? (
              <label className="flex items-center gap-1.5 text-sm">
                <Checkbox
                  checked={!!options.logY}
                  onCheckedChange={(v) => setOption("logY", v === true)}
                />
                Log Y
              </label>
            ) : null}
            {kind === "bar" ? (
              <>
                <label className="flex items-center gap-1.5 text-sm">
                  <Checkbox
                    checked={!!options.horizontal}
                    onCheckedChange={(v) => setOption("horizontal", v === true)}
                  />
                  Horizontal
                </label>
                <label className="flex items-center gap-1.5 text-sm">
                  <Checkbox
                    checked={!!options.stacked}
                    onCheckedChange={(v) => setOption("stacked", v === true)}
                  />
                  Stacked
                </label>
              </>
            ) : null}
            {kind === "line" ? (
              <label className="flex items-center gap-1.5 text-sm">
                <Checkbox
                  checked={!!options.step}
                  onCheckedChange={(v) => setOption("step", v === true)}
                />
                Step
              </label>
            ) : null}
            {kind !== "pie" && kind !== "radar" ? (
              <div className="flex items-center gap-1.5">
                <Label htmlFor="insert-chart-error" className="text-sm font-normal">
                  Error bars
                </Label>
                <Select
                  value={options.errorColumn ?? "none"}
                  onValueChange={(v) => setOption("errorColumn", v === "none" ? null : v)}
                  disabled={numericColumns.length === 0}
                >
                  <SelectTrigger id="insert-chart-error" className="h-8 w-36">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    {numericColumns.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ) : null}
          </div>
        </FieldSet>

        {/* There are only five hue/shape slots, so a 6th group can't be drawn.
            Say so rather than dropping it quietly. */}
        {/* Placeholders show the derived default, so it is obvious that a blank
            field still produces a labelled axis. Radar and pie have no axes. */}
        {refLinesApply || kind === "bar" || kind === "line" ? (
          <div className="grid gap-4 sm:grid-cols-2">
            <Field>
              <FieldLabel htmlFor="insert-chart-xlabel">X axis title</FieldLabel>
              <Input
                id="insert-chart-xlabel"
                className="h-8"
                value={options.xLabel ?? ""}
                placeholder={titles.x || "none"}
                onChange={(e) => setOption("xLabel", e.target.value || null)}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="insert-chart-ylabel">Y axis title</FieldLabel>
              <Input
                id="insert-chart-ylabel"
                className="h-8"
                value={options.yLabel ?? ""}
                placeholder={titles.y || "none (the legend names the series)"}
                onChange={(e) => setOption("yLabel", e.target.value || null)}
              />
            </Field>
          </div>
        ) : null}

        {refLinesApply ? (
          <FieldSet>
            <FieldLegend variant="label">Reference lines</FieldLegend>
            <div className="flex flex-col gap-2">
              {segmentsApply ? (
                <div className="flex flex-wrap items-center gap-4">
                  {SEGMENTS.map((seg) => (
                    <label key={seg.value} className="flex items-center gap-1.5 text-sm">
                      <Checkbox
                        checked={segments.some((s2) => s2.segment === seg.value)}
                        onCheckedChange={(v) =>
                          writeRefLines(
                            v === true
                              ? [...segments, { segment: seg.value }]
                              : segments.filter((s2) => s2.segment !== seg.value),
                            valueLines,
                          )
                        }
                      />
                      {seg.label}
                    </label>
                  ))}
                </div>
              ) : null}

              {/* Capped height with its own scroller: adding a fourth line must
                  not keep growing the dialog (no layout jump). */}
              {valueLines.length > 0 ? (
                <div className="flex max-h-28 flex-col gap-2 overflow-y-auto">
                  {valueLines.map((line, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Select
                        value={line.axis}
                        onValueChange={(v) =>
                          writeRefLines(
                            segments,
                            valueLines.map((l, j) =>
                              j === i ? { ...l, axis: v as "x" | "y" } : l,
                            ),
                          )
                        }
                      >
                        <SelectTrigger className="h-8 w-16" aria-label="Axis">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {xRangeApplies ? <SelectItem value="x">X</SelectItem> : null}
                          {yRangeApplies ? <SelectItem value="y">Y</SelectItem> : null}
                        </SelectContent>
                      </Select>
                      <Input
                        type="number"
                        className="h-8 w-24"
                        aria-label="Value"
                        value={line.value}
                        onChange={(e) =>
                          writeRefLines(
                            segments,
                            valueLines.map((l, j) =>
                              j === i ? { ...l, value: Number(e.target.value) } : l,
                            ),
                          )
                        }
                      />
                      <Input
                        className="h-8 flex-1"
                        aria-label="Label"
                        placeholder="label (optional)"
                        value={line.label ?? ""}
                        onChange={(e) =>
                          writeRefLines(
                            segments,
                            valueLines.map((l, j) =>
                              j === i ? { ...l, label: e.target.value || undefined } : l,
                            ),
                          )
                        }
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Remove reference line"
                        onClick={() =>
                          writeRefLines(
                            segments,
                            valueLines.filter((_, j) => j !== i),
                          )
                        }
                      >
                        <Trash2 />
                      </Button>
                    </div>
                  ))}
                </div>
              ) : null}

              <div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    writeRefLines(segments, [
                      ...valueLines,
                      { axis: yRangeApplies ? "y" : "x", value: 0 },
                    ])
                  }
                >
                  Add line
                </Button>
              </div>
            </div>
          </FieldSet>
        ) : null}

        {/* Always rendered, disabled where it doesn't apply, so switching chart
            type never resizes the dialog (no layout jump). A blank box means
            "fit to the data" — the axis is only pinned once you type a number. */}
        <FieldSet>
          <FieldLegend variant="label">Axis range (blank = fit to data)</FieldLegend>
          <div className="flex flex-wrap items-center gap-2">
            {AXIS_BOUNDS.map((b) => (
              <div key={b.key} className="flex items-center gap-1.5">
                <Label htmlFor={`insert-chart-${b.key}`} className="text-sm font-normal">
                  {b.label}
                </Label>
                <Input
                  id={`insert-chart-${b.key}`}
                  type="number"
                  className="h-8 w-24"
                  placeholder="auto"
                  disabled={b.axis === "x" ? !xRangeApplies : !yRangeApplies}
                  value={options[b.key] ?? ""}
                  onChange={(e) =>
                    setOption(b.key, e.target.value === "" ? null : Number(e.target.value))
                  }
                />
              </div>
            ))}
          </div>
        </FieldSet>

        {/* One template string rather than interleaved JSX expressions: JSX
            strips the whitespace around an expression at a line break, which
            silently produced "the first 5are plotted". */}
        {groupCount > MAX_SERIES ? (
          <p className="text-sm text-warning">
            {`${options.groupBy} has ${groupCount} values, but there are only ${MAX_SERIES} colour and shape slots, so the first ${MAX_SERIES} are plotted. Every row still appears in the chart's data table. To show them all, split the data across several charts or group the rarer values together first.`}
          </p>
        ) : null}

        <Field>
          <FieldLabel htmlFor="insert-chart-caption">Caption (optional)</FieldLabel>
          <Input
            id="insert-chart-caption"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="What this chart shows"
          />
        </Field>
      </div>
    </EmbedDialogFrame>
  );
}
