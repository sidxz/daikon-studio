/**
 * Amino acids grouped by the property that decides what a substitution does.
 *
 * Named, not only colored. The standard palettes a protein scientist reads (Clustal,
 * Taylor, Zappo) encode exactly these groups, but they encode them *only* as hue --
 * which assumes the reader has the key memorised and leaves everyone else guessing.
 * This app's copy rule is plain language over jargon, so a substitution says
 * "hydrophobic → basic" in words and uses color to make the change scannable.
 */
export type ResidueProperty = "hydrophobic" | "polar" | "basic" | "acidic" | "special";

const PROPERTY_OF: Record<string, ResidueProperty> = {
  A: "hydrophobic",
  V: "hydrophobic",
  L: "hydrophobic",
  I: "hydrophobic",
  M: "hydrophobic",
  F: "hydrophobic",
  W: "hydrophobic",
  C: "hydrophobic",
  S: "polar",
  T: "polar",
  N: "polar",
  Q: "polar",
  Y: "polar",
  K: "basic",
  R: "basic",
  H: "basic",
  D: "acidic",
  E: "acidic",
  // Glycine and proline are grouped apart because they are backbone effects rather
  // than side-chain ones: glycine adds flexibility and proline removes it, and either
  // can matter more than whatever the side chain would have done.
  G: "special",
  P: "special",
};

export function residueProperty(residue: string): ResidueProperty | undefined {
  return PROPERTY_OF[residue.toUpperCase()];
}

/** Tailwind classes per property. Readable on both themes; see `globals.css` tokens. */
export const PROPERTY_CLASS: Record<ResidueProperty, string> = {
  hydrophobic: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  polar: "bg-teal-500/15 text-teal-700 dark:text-teal-300",
  basic: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  acidic: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
  special: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
};

/** The 20 standard residues, grouped so a map's axis reads by property, not alphabet. */
export const RESIDUE_AXIS = [..."AVLIMFWC", ..."STNQY", ..."KRH", ..."DE", ..."GP"] as const;
