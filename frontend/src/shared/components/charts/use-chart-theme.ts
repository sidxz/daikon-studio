"use client";

import { useEffect, useState } from "react";

/**
 * The design tokens' chart colours, resolved to concrete values.
 *
 * Plot writes most colours straight onto SVG attributes, where `var(--chart-1)`
 * resolves fine -- but a *sequential* scale has to interpolate between its
 * endpoints, and d3 cannot interpolate a string it has not resolved. So the
 * tokens are read off the document once per theme rather than passed through as
 * `var()`, which also keeps every chart on the same values as the rest of the UI
 * instead of a second hardcoded palette drifting beside the first.
 *
 * Re-read when `data-theme` changes, because dark mode here is a *selected*
 * palette from the tokens package and not a filter over the light one.
 */
export interface ChartTheme {
  /** Two series. Blue and amber. */
  pair: [string, string];
  /** Three series. The pair, plus teal. */
  triple: [string, string, string];
  /** One hue for a sequential ramp, light end to dark end. */
  sequential: [string, string];
  text: string;
  muted: string;
  grid: string;
  surface: string;
  warning: string;
  success: string;
  /** Held-out and new compounds (test set, a run's compounds): amber, darker in light mode. */
  held: string;
}

const TOKENS = {
  blue: "--chart-1",
  teal: "--chart-2",
  violet: "--chart-3",
  amber: "--chart-4",
  red: "--chart-5",
  text: "--color-foreground",
  muted: "--color-muted-foreground",
  border: "--color-border",
  surface: "--color-card",
  warning: "--color-ds-warning",
  success: "--color-ds-success",
  held: "--color-score-fair",
} as const;

function read(): ChartTheme {
  const style = getComputedStyle(document.documentElement);
  const get = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;

  const blue = get(TOKENS.blue, "#3b82f6");
  const amber = get(TOKENS.amber, "#f59e0b");
  const teal = get(TOKENS.teal, "#14b8a6");
  const muted = get(TOKENS.muted, "#71717a");

  return {
    // Blue and amber, and deliberately not blue and violet: measured against
    // this token set, blue↔violet separate by ΔE 1.3 under deuteranopia and
    // 12.0 with full colour vision -- indistinguishable for roughly 1 in 12 men
    // and hard for everyone else, despite looking obviously different to the
    // person choosing them. Blue↔amber measures ΔE 32.8 and 38.0.
    pair: [blue, amber],
    triple: [blue, amber, teal],
    // One hue, pale to saturated: a magnitude has an order, and only lightness
    // within a single hue carries an order that survives being printed,
    // photocopied or seen by a colourblind reader.
    sequential: [get(TOKENS.surface, "#ffffff"), blue],
    text: get(TOKENS.text, "#18181b"),
    muted,
    grid: get(TOKENS.border, "#e4e4e7"),
    surface: get(TOKENS.surface, "#ffffff"),
    warning: get(TOKENS.warning, "#f59e0b"),
    success: get(TOKENS.success, "#22c55e"),
    held: get(TOKENS.held, "#d97706"),
  };
}

export function useChartTheme(): ChartTheme | null {
  // Null until mounted: `getComputedStyle` needs a document, and rendering a
  // chart with guessed colours server-side would flash the wrong palette.
  const [theme, setTheme] = useState<ChartTheme | null>(null);

  useEffect(() => {
    setTheme(read());
    const observer = new MutationObserver(() => setTheme(read()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme", "class"],
    });
    return () => observer.disconnect();
  }, []);

  return theme;
}
