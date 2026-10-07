import type { SplitStrategy } from "@/shared/lib/api/model";

/**
 * What each split actually held out, in the words a reader is shown.
 *
 * Every strategy but `random` is a *grouped* split: a set of related rows is
 * kept entirely on one side, and they differ only in what "related" means. The
 * UI therefore varies one thing -- this vocabulary -- rather than branching on
 * the strategy name in a dozen components, each with its own idea of what was
 * held out. Keyed by strategy so a fifth split cannot be added without
 * supplying the words for it.
 */
export interface SplitVocabulary {
  /** The thing kept whole, singular, as a reader is told it. `null` for a random split. */
  group: string | null;
  /** Completes "scored on a <name> split: ..." -- what was held out, truthfully. */
  held: string;
  /** Completes "Performance on ..." over the optimism gap. */
  unfamiliar: string;
}

export const SPLIT_VOCABULARY: Record<SplitStrategy, SplitVocabulary> = {
  random: {
    group: null,
    held: "the test set contains close relatives of the training data, so scores are likely optimistic",
    unfamiliar: "unfamiliar chemistry",
  },
  scaffold: {
    group: "Bemis–Murcko scaffold",
    held: "no test scaffold appears in the training set",
    unfamiliar: "unfamiliar chemistry",
  },
  identity: {
    group: "protein family",
    held: "no protein family in the test set appears in the training set",
    unfamiliar: "unfamiliar proteins",
  },
  position: {
    group: "mutated residue position",
    held: "no residue position mutated in the test set was mutated in training",
    unfamiliar: "untested positions",
  },
};

/**
 * `ScorecardResponse.split_strategy` is a bare `string` on the wire, so a
 * strategy this build has never heard of is representable. Saying nothing about
 * it is correct; printing another split's sentence is the bug this whole module
 * exists to prevent.
 */
const UNKNOWN: SplitVocabulary = {
  group: "group of related rows",
  held: "related rows are kept to one side of the split",
  unfamiliar: "unfamiliar data",
};

export function splitVocabulary(strategy: string): SplitVocabulary {
  return SPLIT_VOCABULARY[strategy as SplitStrategy] ?? UNKNOWN;
}

/** Random is the only split that leaves related rows on both sides. */
export function isGroupedSplit(strategy: string): boolean {
  return splitVocabulary(strategy).group !== null;
}

/** "scaffold" -> "Scaffold", for the places that need a label and not a sentence. */
export function splitTitle(strategy: string): string {
  return strategy.charAt(0).toUpperCase() + strategy.slice(1);
}
