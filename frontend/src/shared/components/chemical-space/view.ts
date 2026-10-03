/** The map's camera and hit testing. Map coordinates are the API's [0, 1] square, y up. */
export interface View {
  cx: number;
  cy: number;
  scale: number;
}

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 64;

/** `[minX, minY, maxX, maxY]` of the points, in map units. */
export type Bounds = [number, number, number, number];

/**
 * Fit the data's own bounds into the canvas, so a wide card is filled rather
 * than holding a centred square. Without bounds (or for a single point) the
 * unit square is fitted.
 */
export function fitView(width: number, height: number, padding = 16, bounds?: Bounds): View {
  const unit = { cx: 0.5, cy: 0.5, scale: Math.max(1, Math.min(width, height) - padding * 2) };
  if (!bounds) return unit;
  const [minX, minY, maxX, maxY] = bounds;
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  if (spanX <= 0 && spanY <= 0) return unit;
  const scale = Math.min(
    spanX > 0 ? (width - padding * 2) / spanX : Number.POSITIVE_INFINITY,
    spanY > 0 ? (height - padding * 2) / spanY : Number.POSITIVE_INFINITY,
  );
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, scale: Math.max(1, scale) };
}

/** Bounds over every point of the given layers, without spreading large arrays. */
export function boundsOf(...layers: { x: ArrayLike<number>; y: ArrayLike<number> }[]): Bounds {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const layer of layers) {
    for (let i = 0; i < layer.x.length; i++) {
      minX = Math.min(minX, layer.x[i]);
      maxX = Math.max(maxX, layer.x[i]);
      minY = Math.min(minY, layer.y[i]);
      maxY = Math.max(maxY, layer.y[i]);
    }
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : [0, 0, 1, 1];
}

export function toScreen(
  view: View,
  width: number,
  height: number,
  x: number,
  y: number,
): [number, number] {
  return [(x - view.cx) * view.scale + width / 2, height / 2 - (y - view.cy) * view.scale];
}

export function toMap(
  view: View,
  width: number,
  height: number,
  sx: number,
  sy: number,
): [number, number] {
  return [(sx - width / 2) / view.scale + view.cx, view.cy - (sy - height / 2) / view.scale];
}

export function zoomAt(
  view: View,
  width: number,
  height: number,
  sx: number,
  sy: number,
  factor: number,
  fit: View,
): View {
  const [mx, my] = toMap(view, width, height, sx, sy);
  const scale = Math.min(fit.scale * MAX_ZOOM, Math.max(fit.scale * MIN_ZOOM, view.scale * factor));
  return { scale, cx: mx - (sx - width / 2) / scale, cy: my + (sy - height / 2) / scale };
}

export function pan(view: View, dx: number, dy: number): View {
  return { ...view, cx: view.cx - dx / view.scale, cy: view.cy + dy / view.scale };
}

export interface PickGrid {
  size: number;
  cells: Map<number, number[]>;
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
}

const cellOf = (v: number, size: number) => Math.min(size - 1, Math.max(0, Math.floor(v * size)));

/** A uniform grid over the unit square: hovering 100k points stays O(points near the cursor). */
export function buildPickGrid(xs: ArrayLike<number>, ys: ArrayLike<number>, size = 128): PickGrid {
  const cells = new Map<number, number[]>();
  for (let i = 0; i < xs.length; i++) {
    const key = cellOf(xs[i], size) * size + cellOf(ys[i], size);
    const cell = cells.get(key);
    if (cell) cell.push(i);
    else cells.set(key, [i]);
  }
  return { size, cells, xs, ys };
}

/** Index of the nearest point within `radius` (map units), or -1. */
export function pickNearest(grid: PickGrid, x: number, y: number, radius: number): number {
  const reach = Math.ceil(radius * grid.size);
  const col = cellOf(x, grid.size);
  const row = cellOf(y, grid.size);
  let best = -1;
  let bestDistance = radius * radius;
  for (let c = Math.max(0, col - reach); c <= Math.min(grid.size - 1, col + reach); c++) {
    for (let r = Math.max(0, row - reach); r <= Math.min(grid.size - 1, row + reach); r++) {
      for (const i of grid.cells.get(c * grid.size + r) ?? []) {
        const d = (grid.xs[i] - x) ** 2 + (grid.ys[i] - y) ** 2;
        if (d <= bestDistance) {
          bestDistance = d;
          best = i;
        }
      }
    }
  }
  return best;
}
