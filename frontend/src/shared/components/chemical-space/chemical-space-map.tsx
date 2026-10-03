"use client";

import { useChartTheme } from "@/shared/components/charts/use-chart-theme";
import { cn } from "@/shared/lib/utils";
import {
  type PointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { type Rgba, parseColor } from "./color";
import { type MapRenderer, createMapRenderer } from "./renderer";
import {
  type View,
  boundsOf,
  buildPickGrid,
  fitView,
  pan,
  pickNearest,
  toMap,
  toScreen,
  zoomAt,
} from "./view";

/** Parallel arrays in the API's unit square; `style` is a `STYLE` id per point. */
export interface MapLayer {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  style: ArrayLike<number>;
}

export interface MapHit {
  layer: "base" | "overlay";
  index: number;
}

interface ChemicalSpaceMapProps {
  base: MapLayer;
  overlay?: MapLayer;
  /** By `STYLE` id. Null until the theme is read; nothing is drawn until then. */
  colors: Rgba[] | null;
  /** Accessible summary of what the map shows, with real counts. */
  label: string;
  pickBase?: boolean;
  /** Map-space endpoints joined to the hovered point, e.g. its nearest training compounds. */
  lines?: (hit: MapHit) => [number, number][];
  tooltip?: (hit: MapHit) => ReactNode;
  className?: string;
}

const PICK_RADIUS_PX = 10;
const ENTRANCE_MS = 900;
const TOOLTIP = { width: 232, offset: 14 };

/** The map's palette from the design tokens, re-read when the theme changes. */
export function useMapColors(): Rgba[] | null {
  const theme = useChartTheme();
  return useMemo(
    () =>
      theme && [
        parseColor(theme.pair[0], 0.6), // training
        parseColor(theme.triple[2], 0.7), // validation
        parseColor(theme.held, 0.85), // test
        parseColor(theme.held), // run, inside the domain
        parseColor(theme.held), // run, outside the domain (dashed ring)
      ],
    [theme],
  );
}

function interleave(layer: MapLayer): [Float32Array, Float32Array] {
  const positions = new Float32Array(layer.x.length * 2);
  const styles = new Float32Array(layer.x.length);
  for (let i = 0; i < layer.x.length; i++) {
    positions[i * 2] = layer.x[i];
    positions[i * 2 + 1] = layer.y[i];
    styles[i] = layer.style[i];
  }
  return [positions, styles];
}

/**
 * Where the hover box goes: beside the cursor, flipped toward the map's centre in
 * each axis, so a compound near the bottom or right edge opens its box upward or
 * leftward and the last line (the similarity) is never cut off. The box may extend
 * past the map's frame; it is drawn outside the clipped canvas area.
 */
export function tooltipPlacement(
  sx: number,
  sy: number,
  width: number,
  height: number,
): { left: number; top: number; transform: string } {
  const flipX = sx > width / 2;
  const flipY = sy > height / 2;
  return {
    left: flipX ? sx - TOOLTIP.offset : sx + TOOLTIP.offset,
    top: flipY ? sy - TOOLTIP.offset : sy + TOOLTIP.offset,
    transform: `translate(${flipX ? "-100%" : "0"}, ${flipY ? "-100%" : "0"})`,
  };
}

/**
 * The interactive map. WebGL draws the points; a thin SVG overlay draws the
 * hover lines, and a tooltip box shows whatever the caller renders for a hit.
 * Drag pans, pinch or Ctrl/⌘ + scroll zooms (plain scroll keeps scrolling the
 * page), double-click resets.
 */
export function ChemicalSpaceMap({
  base,
  overlay,
  colors,
  label,
  pickBase = false,
  lines,
  tooltip,
  className,
}: ChemicalSpaceMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<MapRenderer | null>(null);
  const viewRef = useRef<View>({ cx: 0.5, cy: 0.5, scale: 1 });
  const fitRef = useRef<View>(viewRef.current);
  const sizeRef = useRef<[number, number]>([0, 0]);
  const boundsRef = useRef(boundsOf(base, ...(overlay ? [overlay] : [])));
  const progressRef = useRef(1);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [hover, setHover] = useState<(MapHit & { sx: number; sy: number }) | null>(null);
  const [dragging, setDragging] = useState(false);
  // Bumped on pan and zoom so the SVG overlay re-projects; the canvas redraws itself.
  const [, setViewTick] = useState(0);

  const draw = useCallback(() => {
    rendererRef.current?.draw(viewRef.current, progressRef.current);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = createMapRenderer(canvas);
    if (!renderer) {
      setUnsupported(true);
      return;
    }
    rendererRef.current = renderer;
    const observer = new ResizeObserver(() => {
      const [width, height] = renderer.resize();
      const first = sizeRef.current[0] === 0;
      sizeRef.current = [width, height];
      fitRef.current = fitView(width, height, 24, boundsRef.current);
      if (first) viewRef.current = fitRef.current;
      draw();
    });
    observer.observe(canvas);

    let frame = 0;
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      progressRef.current = 0;
      let start: number | null = null;
      const step = (now: number) => {
        start ??= now;
        progressRef.current = Math.min(1, (now - start) / ENTRANCE_MS);
        draw();
        if (progressRef.current < 1) frame = requestAnimationFrame(step);
      };
      frame = requestAnimationFrame(step);
    }
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [draw]);

  useEffect(() => {
    const [positions, styles] = interleave(base);
    rendererRef.current?.setLayer(0, positions, styles);
    // A few hundred compounds read as dust at 2.5 px; 100k need small dots to stay legible.
    rendererRef.current?.setPointScale(
      base.x.length <= 2_000 ? 1.6 : base.x.length <= 20_000 ? 1.2 : 0.8,
    );
    draw();
  }, [base, draw]);

  useEffect(() => {
    // Refit when the data changes; the viewer's pan and zoom survive only within one dataset.
    boundsRef.current = boundsOf(base, ...(overlay ? [overlay] : []));
    const [width, height] = sizeRef.current;
    if (width === 0) return;
    fitRef.current = fitView(width, height, 24, boundsRef.current);
    viewRef.current = fitRef.current;
    draw();
  }, [base, overlay, draw]);

  useEffect(() => {
    const [positions, styles] = overlay
      ? interleave(overlay)
      : [new Float32Array(0), new Float32Array(0)];
    rendererRef.current?.setLayer(1, positions, styles);
    draw();
  }, [overlay, draw]);

  useEffect(() => {
    if (!colors) return;
    rendererRef.current?.setColors(colors);
    draw();
  }, [colors, draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      // Pinch arrives as a wheel with ctrlKey; a plain wheel scrolls the page.
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const [width, height] = sizeRef.current;
      viewRef.current = zoomAt(
        viewRef.current,
        width,
        height,
        event.clientX - rect.left,
        event.clientY - rect.top,
        Math.exp(-event.deltaY * 0.01),
        fitRef.current,
      );
      draw();
      setHover(null);
      setViewTick((tick) => tick + 1);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [draw]);

  const baseGrid = useMemo(
    () => (pickBase ? buildPickGrid(base.x, base.y) : null),
    [base, pickBase],
  );
  const overlayGrid = useMemo(
    () => (overlay ? buildPickGrid(overlay.x, overlay.y) : null),
    [overlay],
  );

  const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { x: event.clientX, y: event.clientY };
    setDragging(true);
    setHover(null);
  };

  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current) {
      viewRef.current = pan(
        viewRef.current,
        event.clientX - dragRef.current.x,
        event.clientY - dragRef.current.y,
      );
      dragRef.current = { x: event.clientX, y: event.clientY };
      draw();
      setViewTick((tick) => tick + 1);
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    const [width, height] = sizeRef.current;
    const [mx, my] = toMap(viewRef.current, width, height, sx, sy);
    const radius = PICK_RADIUS_PX / viewRef.current.scale;
    let hit: MapHit | null = null;
    if (overlayGrid) {
      const index = pickNearest(overlayGrid, mx, my, radius);
      if (index >= 0) hit = { layer: "overlay", index };
    }
    if (!hit && baseGrid) {
      const index = pickNearest(baseGrid, mx, my, radius);
      if (index >= 0) hit = { layer: "base", index };
    }
    setHover(hit && { ...hit, sx, sy });
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragging(false);
  };

  const reset = () => {
    viewRef.current = fitRef.current;
    draw();
    setViewTick((tick) => tick + 1);
  };

  if (unsupported) {
    return <p className="text-sm text-muted-foreground">This browser cannot draw the map.</p>;
  }

  const [width, height] = sizeRef.current;
  const hovered = hover && (hover.layer === "overlay" ? overlay : base);
  const anchor =
    hover && hovered
      ? toScreen(viewRef.current, width, height, hovered.x[hover.index], hovered.y[hover.index])
      : null;
  const ends = hover && lines ? lines(hover) : [];

  return (
    <div className={cn("relative", className)}>
      <div className="relative h-[420px] w-full overflow-hidden rounded-lg border bg-card">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={label}
          className={cn("block size-full touch-none", dragging ? "cursor-grabbing" : "cursor-grab")}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onPointerLeave={() => {
            if (!dragRef.current) setHover(null);
          }}
          onDoubleClick={reset}
        />
        {anchor && ends.length > 0 && (
          <svg
            className="pointer-events-none absolute inset-0 size-full text-foreground/60"
            aria-hidden="true"
          >
            {ends.map(([x, y]) => {
              const [ex, ey] = toScreen(viewRef.current, width, height, x, y);
              return (
                <g key={`${x}:${y}`}>
                  <line
                    x1={anchor[0]}
                    y1={anchor[1]}
                    x2={ex}
                    y2={ey}
                    stroke="currentColor"
                    strokeWidth={1}
                  />
                  <circle
                    cx={ex}
                    cy={ey}
                    r={4}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  />
                </g>
              );
            })}
          </svg>
        )}
        <p className="pointer-events-none absolute bottom-2 left-3 text-[11px] text-muted-foreground">
          Drag to pan · pinch or Ctrl/⌘ + scroll to zoom · double-click to reset
        </p>
      </div>
      {hover && tooltip && (
        <div
          className="pointer-events-none absolute z-20 rounded-md border bg-popover p-2 text-xs text-popover-foreground shadow-md"
          style={{ width: TOOLTIP.width, ...tooltipPlacement(hover.sx, hover.sy, width, height) }}
        >
          {tooltip(hover)}
        </div>
      )}
    </div>
  );
}
