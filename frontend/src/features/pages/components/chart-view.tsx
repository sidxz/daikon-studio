"use client";

import type { NodeViewProps } from "@tiptap/react";
import dynamic from "next/dynamic";
import { useMemo } from "react";

import { parseTable, resolveColumns } from "@/features/pages/lib/parse-table";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";
import type { ChartKind, ChartOptions } from "./renderers/chart-figure";

// Recharts is ~100KB gzipped and most pages have no chart, so it loads only
// when one is on screen. ssr:false matches the editor's immediatelyRender:false.
const ChartFigure = dynamic(() => import("./renderers/chart-figure").then((m) => m.ChartFigure), {
  ssr: false,
  loading: () => <div className="h-64 animate-pulse rounded-md bg-muted/30" />,
});

const KIND_LABEL: Record<ChartKind, string> = {
  bar: "Bar chart",
  line: "Line chart",
  scatter: "Scatter plot",
  radar: "Radar chart",
  pie: "Pie chart",
};

export function ChartView(props: NodeViewProps) {
  const { kind, raw, x, series, options, caption, width } = props.node.attrs as {
    kind: ChartKind;
    raw: string;
    x: string | null;
    series: string[] | null;
    options: ChartOptions;
    caption: string | null;
    width: EmbedWidth | null;
  };

  const table = useMemo(() => parseTable(raw), [raw]);
  const roles = useMemo(() => resolveColumns(table, x, series), [table, x, series]);

  return (
    <EmbedFigure {...props} kind="chart" width={width ?? "full"}>
      <ChartFigure kind={kind} raw={raw} x={x} series={series} options={options} />

      <figcaption className="mt-1 text-sm">
        {caption ? <span className="text-foreground">{caption}</span> : null}

        {/* The light-mode palette's teal and amber sit below 3:1 against a white
            surface, which obligates a non-color reading of the data. This
            disclosure is that relief — and in a lab notebook the underlying
            numbers are worth having anyway. Collapsed by default so it costs
            nothing visually; <details> means no JS and no layout jump above it. */}
        {table.rows.length > 0 ? (
          <details className="mt-1">
            <summary className="cursor-pointer text-muted-foreground select-none">
              Show data ({table.rows.length} rows · {KIND_LABEL[kind] ?? "Chart"})
            </summary>
            {/* Own scroller: a wide table must never make the page scroll sideways. */}
            <div className="mt-2 max-h-64 overflow-auto rounded-md border border-border">
              <table className="w-full text-left text-xs">
                <thead className="bg-muted/50 text-muted-foreground">
                  <tr>
                    {table.columns.map((c) => (
                      <th key={c} scope="col" className="px-2 py-1 font-medium">
                        {c}
                        {c === roles.x ? " (x)" : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  {table.rows.map((row, i) => (
                    <tr key={i} className="border-t border-border">
                      {table.columns.map((c) => (
                        <td key={c} className="px-2 py-1">
                          {String(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ) : null}
      </figcaption>
    </EmbedFigure>
  );
}
