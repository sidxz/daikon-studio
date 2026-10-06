import { themeQuartz } from "ag-grid-community";

/**
 * AG Grid wired to the design tokens rather than its own palette, so it follows
 * the `data-theme` switch without a second theme definition or a re-render.
 */
export const studioGridTheme = themeQuartz.withParams({
  backgroundColor: "var(--background)",
  foregroundColor: "var(--foreground)",
  borderColor: "var(--border)",
  headerBackgroundColor: "var(--muted)",
  headerTextColor: "var(--foreground)",
  headerFontWeight: 500,
  rowHoverColor: "color-mix(in oklch, var(--muted) 50%, transparent)",
  selectedRowBackgroundColor: "var(--accent)",
  oddRowBackgroundColor: "transparent",
  cellTextColor: "var(--foreground)",
  fontFamily: "var(--font-sans)",
  fontSize: 13,
  headerFontSize: 12,
  rowBorder: { color: "var(--border)", style: "solid", width: 1 },
  columnBorder: false,
  wrapperBorderRadius: "var(--radius-lg)",
  wrapperBorder: { color: "var(--border)", style: "solid", width: 1 },
  spacing: 6,
});

/** Results have a richer header while retaining the shared body and theme tokens. */
export const resultsGridTheme = studioGridTheme.withParams({
  headerBackgroundColor: "var(--ds-surface-elevated)",
  headerFontWeight: 600,
  headerCellHoverBackgroundColor: "var(--ds-accent-subtle)",
  headerCellMovingBackgroundColor: "var(--ds-accent-light)",
  headerColumnBorder: false,
  headerRowBorder: { color: "var(--ds-border-subtle)", style: "solid", width: 1 },
  headerColumnResizeHandleColor: "var(--ds-border-subtle)",
  headerColumnResizeHandleHeight: "24%",
  headerColumnResizeHandleWidth: 1,
});
