/** How many structures the preview draws. Enough to recognise the set. */
export const PREVIEW_SAMPLE_SIZE = 6;

export interface PreviewSummary {
  /** Every data row in the file, blank structure cells included. */
  total: number;
  /** The first few non-blank structures, for thumbnails. */
  sample: string[];
  /** Rows whose structure cell is empty. */
  blank: number;
}

/**
 * What a dropped CSV contains, before anything is uploaded.
 *
 * Counting happens here rather than in the component so the number on the Run
 * button -- the one the user is asked to commit to -- is covered by a test.
 */
export function summarisePreview(
  rows: Record<string, string | undefined>[],
  column: string,
): PreviewSummary {
  const sample: string[] = [];
  let blank = 0;

  for (const row of rows) {
    const value = row[column]?.trim() ?? "";
    if (value === "") {
      blank += 1;
      continue;
    }
    if (sample.length < PREVIEW_SAMPLE_SIZE) sample.push(value);
  }

  return { total: rows.length, sample, blank };
}
