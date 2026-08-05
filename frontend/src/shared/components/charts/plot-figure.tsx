"use client";

import { cn } from "@/shared/lib/utils";
import * as Plot from "@observablehq/plot";
import { useEffect, useRef, useState } from "react";

/**
 * The one place Observable Plot touches the DOM.
 *
 * Plot is imperative -- `Plot.plot(options)` returns an SVG element rather than
 * describing one -- so exactly one component owns the append/replace/remove
 * cycle and every chart in the product is a `options => spec` function with no
 * lifecycle of its own. That keeps the charts themselves pure and testable, and
 * it means a leak or a double-render can only be wrong in one file.
 *
 * Width comes from a ResizeObserver rather than CSS: Plot renders a fixed-width
 * SVG, so a chart that is not re-plotted on resize is either clipped or leaves a
 * gap. Height stays a prop -- these are dense diagnostic charts and letting them
 * pick their own height makes a page of them ragged.
 */
export function PlotFigure({
  options,
  height = 220,
  className,
  caption,
}: {
  options: (width: number) => Plot.PlotOptions;
  height?: number;
  className?: string;
  caption?: React.ReactNode;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const element = container.current;
    if (!element || width === 0) return;
    const chart = Plot.plot({ ...options(width), width, height });
    element.append(chart);
    // Plot builds a fresh element per render; without this the old one stays in
    // the DOM and every re-render stacks another chart on top of the last.
    return () => chart.remove();
  }, [options, width, height]);

  return (
    <figure className={cn("w-full", className)}>
      <div ref={container} className="w-full overflow-x-auto" />
      {caption && (
        <figcaption className="mt-1.5 text-xs text-muted-foreground">{caption}</figcaption>
      )}
    </figure>
  );
}

/**
 * Shared Plot options: recessive axes, no chart-drawn background, tabular
 * numbers. Spread into every chart so a change lands everywhere at once.
 */
export function baseOptions(theme: {
  text: string;
  muted: string;
  grid: string;
}): Partial<Plot.PlotOptions> {
  return {
    style: {
      background: "transparent",
      color: theme.muted,
      fontSize: "11px",
      fontVariantNumeric: "tabular-nums",
      overflow: "visible",
    },
    marginLeft: 48,
    marginBottom: 34,
    // Plot writes the y-axis label above the axis, in the top margin. At 12 it
    // lands on top of the highest tick label on every chart that has one.
    marginTop: 22,
    marginRight: 12,
  };
}

/**
 * A legend row. Plot can draw its own swatches, but they arrive as a separate
 * element with their own font stack, and the pages here need the legend inside
 * the card header where the caption is -- so identity is never carried by
 * colour alone even when a chart scrolls out of view.
 */
export function ChartLegend({
  items,
  className,
}: {
  items: { label: string; color: string }[];
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-center gap-x-4 gap-y-1", className)}>
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span aria-hidden className="size-2.5 rounded-[2px]" style={{ background: item.color }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}
