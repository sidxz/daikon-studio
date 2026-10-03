export type Rgba = [number, number, number, number];

/** A resolved CSS color (hex or rgb[a]) as 0..1 floats for a WebGL uniform. */
export function parseColor(css: string, alpha = 1): Rgba {
  const value = css.trim();
  if (value.startsWith("#")) {
    const hex =
      value.length === 4 ? [...value.slice(1)].map((c) => c + c).join("") : value.slice(1, 7);
    const n = Number.parseInt(hex, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, alpha];
  }
  const parts = value.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0];
  return [parts[0] / 255, parts[1] / 255, parts[2] / 255, (parts[3] ?? 1) * alpha];
}
