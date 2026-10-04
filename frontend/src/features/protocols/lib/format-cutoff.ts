/**
 * A decision cutoff for display: three significant digits, but never a value that
 * rounds up to 1. Boosted trees tune cutoffs such as 0.99997, and "cutoff 1.0" would
 * read as "nothing is ever called active", contradicting the class column.
 */
export function formatCutoff(value: number): string {
  const rounded = Number(value.toPrecision(3));
  if (rounded < 1 || value >= 1) return String(rounded);
  // Below 1 yet rounding to it: add decimals until the shown value is still below 1.
  // Eight is enough for any float32 probability.
  for (let digits = 3; digits <= 8; digits++) {
    const text = value.toFixed(digits);
    if (Number(text) < 1) return text;
  }
  return String(value);
}
