export type Rgb = { r: number; g: number; b: number };

const WHITE: Rgb = { r: 255, g: 255, b: 255 };

/** Parse the colour forms the design tokens actually produce — `#rgb`, `#rrggbb`,
 *  and `rgb()/rgba()`. Exported for its unit test. */
export function parseCssColor(raw: string): Rgb | null {
  const value = raw.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value)?.[1];
  if (hex) {
    const full = hex.length === 3 ? [...hex].map((c) => c + c).join("") : hex;
    return {
      r: Number.parseInt(full.slice(0, 2), 16),
      g: Number.parseInt(full.slice(2, 4), 16),
      b: Number.parseInt(full.slice(4, 6), 16),
    };
  }
  const [r, g, b] = /^rgba?\(/i.test(value) ? (value.match(/\d+/g) ?? []).map(Number) : [];
  return r == null || g == null || b == null ? null : { r, g, b };
}

// Mol* paints its own canvas background, so a CSS class can't reach it — the
// colour has to be handed over as numbers. Read the *resolved* --background
// token off the DOM instead of hardcoding light/dark hexes, so a token change
// carries over for free.
export function readBackgroundRgb(): Rgb {
  if (typeof window === "undefined") return WHITE;
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--background");
  return parseCssColor(raw) ?? WHITE;
}
