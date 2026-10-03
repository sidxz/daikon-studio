const ID_PATTERN = /(^|[^a-z])(id|name|identifier|compound|cpd|sample|batch)([^a-z]|$)/i;

/**
 * The upload's likeliest identifier column, for the wizard to preselect: the
 * first header that reads as a name or an id and is not the structures.
 * ponytail: whole words only, so "CompoundID" is missed and the user picks it by hand.
 */
export function guessIdColumn(columns: string[], structureColumn: string): string | null {
  return columns.find((column) => column !== structureColumn && ID_PATTERN.test(column)) ?? null;
}
