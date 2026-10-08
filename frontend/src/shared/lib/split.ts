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
  /**
   * Why there is no random-split comparison, when there is none. Keyed per
   * strategy because the honest sentence differs: a random split *is* the
   * comparison, a grouped split should have had one recorded, and a predefined
   * split has no computed counterpart to compare against at all.
   */
  noGap: string;
}

/** What a grouped split says when its random-split leg is missing from the run. */
const NOT_RECORDED = "No random-split comparison was recorded for this run.";

export const SPLIT_VOCABULARY: Record<SplitStrategy, SplitVocabulary> = {
  random: {
    group: null,
    held: "the test set contains close relatives of the training data, so scores are likely optimistic",
    unfamiliar: "unfamiliar chemistry",
    noGap: "Not applicable: the model was scored on a random split.",
  },
  scaffold: {
    group: "Bemis–Murcko scaffold",
    held: "no test scaffold appears in the training set",
    unfamiliar: "unfamiliar chemistry",
    noGap: NOT_RECORDED,
  },
  identity: {
    group: "protein family",
    held: "no protein family in the test set appears in the training set",
    unfamiliar: "unfamiliar proteins",
    noGap: NOT_RECORDED,
  },
  position: {
    group: "mutated residue position",
    held: "no residue position mutated in the test set was mutated in training",
    unfamiliar: "untested positions",
    noGap: NOT_RECORDED,
  },
  predefined: {
    // Nothing is grouped: the partitions were given, not derived.
    group: null,
    held: "the partitions come from a column in the uploaded file, as the benchmark defined them",
    unfamiliar: "the held-out partition",
    noGap:
      "Not applicable: the partitions came from your file, so there is no computed split to compare against.",
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
  noGap: NOT_RECORDED,
};

export function splitVocabulary(strategy: string): SplitVocabulary {
  return SPLIT_VOCABULARY[strategy as SplitStrategy] ?? UNKNOWN;
}

/** Random is the only split that leaves related rows on both sides. */
export function isGroupedSplit(strategy: string): boolean {
  return splitVocabulary(strategy).group !== null;
}

/**
 * What a shared scaffold between training and test means for this split.
 *
 * Here rather than in the profile view because the view was branching on strategy
 * names, which is the thing this module exists to stop: its `else` arm interpolated
 * `group` straight into "This split groups by ...", and `group` is null for random and
 * predefined splits. That compiles, and prints "groups by null" to a scientist.
 *
 * `leaked` is whether any scaffold is actually shared; `shared` is the sentence
 * counting the compounds, supplied by the caller that has the numbers.
 */
export function scaffoldSeparationNote(
  strategy: string,
  shared: string,
  leaked: boolean,
): string {
  if (strategy === "scaffold") {
    return leaked
      ? `${shared} A scaffold split should prevent this.`
      : "As expected for a scaffold split: the training and test sets share no Bemis–Murcko scaffold.";
  }
  if (strategy === "random") {
    return `${shared} Expected for a random split, and one reason its scores are optimistic.`;
  }
  const group = splitVocabulary(strategy).group;
  if (group === null) {
    // A predefined split separated whatever the benchmark's author decided to
    // separate. We do not know what that was, and guessing would be worse than saying
    // where it came from.
    return `${shared} The partitions came from your file, so whether scaffolds are separated was decided there.`;
  }
  return `${shared} This split groups by ${group}, so it does not separate scaffolds.`;
}

/** "scaffold" -> "Scaffold", for the places that need a label and not a sentence. */
export function splitTitle(strategy: string): string {
  return strategy.charAt(0).toUpperCase() + strategy.slice(1);
}
