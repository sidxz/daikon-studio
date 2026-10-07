"""What a Dataset is *made of* -- the read a scientist needs before training on it.

The ValidationReport next door answers "what did the file contain, and what was
thrown away". This answers the question after it: given that the data survived
validation, is the benchmark it defines an honest one, and what chemistry is in
it? Those are different questions with different failure modes, which is why this
is a separate artifact rather than more fields on the report.

Every number here is derived from the frozen snapshot, so a profile is a pure
function of a `content_hash` and can be cached forever without invalidation --
the same immutability that makes a Dataset citable.

Plain dataclasses only -- no polars/numpy/rdkit, matching `validation.py` and
`domain/execution/scorecard.py`.

A note on what is deliberately *not* here: nothing is scored, ranked or graded.
A dataset of four hundred analogues of one scaffold is not a bad dataset, it is a
congeneric series, and a model trained on it will interpolate well and
extrapolate nowhere. The profile's job is to make that legible before a GPU hour
is spent, not to award it a letter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Histogram:
    """`edges` has exactly one more entry than `counts`: bin *i* spans
    `edges[i]` to `edges[i + 1]`. Bins are carried rather than raw values so the
    payload stays flat regardless of how many compounds produced it, and so
    every consumer draws the same bins instead of choosing its own."""

    edges: list[float]
    counts: list[int]


@dataclass(frozen=True, kw_only=True)
class SplitHistogram:
    """Train and test counts over *shared* edges.

    Shared, not per-partition: two histograms drawn on independently chosen bins
    cannot be overlaid, and overlaying them is the entire point -- a test
    partition sitting in a different part of the range than the train partition
    is a distribution shift that will surface later as bias, and it is invisible
    unless the two are on one axis.
    """

    edges: list[float]
    train: list[int]
    test: list[int]


@dataclass(frozen=True, kw_only=True)
class NumericSummary:
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True, kw_only=True)
class TargetDistribution:
    """The measured target, for a NUMERIC dataset.

    `train`/`test` summaries sit beside the histogram because the single most
    common way a regression metric misleads is a narrow dynamic range: an RMSE of
    0.4 is excellent against a target spanning six log units and meaningless
    against one spanning half of one. The span is only readable from the
    summaries, never from the shape.
    """

    histogram: SplitHistogram
    train: NumericSummary
    test: NumericSummary


@dataclass(frozen=True, kw_only=True)
class ClassBalance:
    """Positives and negatives in one partition, for a BINARY dataset.

    Per partition rather than overall, because a scaffold split can leave a
    dataset that is 30% active overall with a test partition that is 4% active --
    at which point MCC becomes unstable and the honest response is a different
    seed or a different split, not a bigger model.
    """

    split: str
    positive: int
    negative: int


@dataclass(frozen=True, kw_only=True)
class SimilarityProfile:
    """How far the test compounds sit from the training set, before any model.

    This is the Scorecard's `applicability_coverage` asked one step earlier: that
    one reports, after a model has been trained, what fraction of test compounds
    the model had seen anything like. This reports the same geometry from the
    Dataset alone, so the question "is this split actually testing
    generalization?" can be answered before the compute is spent.

    `near_duplicates` is the tail that matters most: test compounds that are
    near-identical to something in the training set are, in effect, already
    known, and every metric computed over them is flattered. Under a random split
    this count is routinely non-zero, which is exactly what a scaffold split
    exists to prevent.
    """

    histogram: Histogram
    median: float
    within_domain: float
    within_domain_threshold: float
    near_duplicates: int
    near_duplicate_threshold: float


@dataclass(frozen=True, kw_only=True)
class ScaffoldEntry:
    smiles: str
    count: int


@dataclass(frozen=True, kw_only=True)
class ScaffoldProfile:
    """Bemis-Murcko scaffold composition.

    `cumulative_coverage[k]` is the fraction of compounds accounted for by the
    `k + 1` most common scaffolds, which is what separates a congeneric series
    (one scaffold, hundreds of analogues, a curve that reaches 1.0 almost
    immediately) from a diverse screening deck (a curve that climbs slowly). The
    distinction changes what any downstream metric is worth, and no single count
    expresses it.

    `cross_split_scaffolds` is an integrity check on the split itself: a scaffold
    split that leaves scaffolds on both sides did not do what it claims. It is
    expected to be non-zero for a random split and zero for a scaffold one, and
    reporting it either way is what makes the claim checkable rather than
    trusted.

    An empty scaffold string means "acyclic" -- `murcko_scaffold` returns `""`
    for a molecule with no ring system. Those compounds are counted two different
    ways here, deliberately, because the two questions are different:

    - In `top`, `unique_count` and `cumulative_coverage` they are one entry.
      "95 compounds have no ring system" is a real fact about the dataset.
    - In `cross_split_scaffolds`/`cross_split_compounds` each is its own group,
      matching what `assign_split` does when it partitions. Two acyclic
      molecules share nothing just because neither has a ring, so a scaffold
      split places them freely on either side and is right to. Counting them as
      one shared family reported a correctly-behaved split as leaking.
    """

    unique_count: int
    singleton_count: int
    largest_fraction: float
    cumulative_coverage: list[float]
    top: list[ScaffoldEntry]
    cross_split_scaffolds: int
    cross_split_compounds: int


@dataclass(frozen=True, kw_only=True)
class DescriptorProfile:
    """One physicochemical property across train and test.

    `target_correlation` is Spearman rank correlation against the target, and it
    is the number worth the most on this page: if a single descriptor correlates
    at 0.85 with the target, a model that beats the mandatory baseline has still
    not proved it learned any chemistry -- it may have learned that descriptor.
    `None` when the correlation is undefined (a constant descriptor, or fewer
    than two comparable values), never a fabricated 0.0, which would read as
    "measured, and unrelated".
    """

    name: str
    histogram: SplitHistogram
    median: float
    target_correlation: float | None


@dataclass(frozen=True, kw_only=True)
class ActivityCliff:
    """Two near-identical structures the assay disagrees about.

    Cliffs are the compounds a model reliably gets wrong, and collectively they
    are a second ceiling on achievable accuracy alongside the assay noise floor:
    no featurization that maps these two structures to nearly the same point can
    predict both. Shown as a pair of structures rather than a count, because the
    only useful response is a chemist looking at what changed.
    """

    left_structure: str
    right_structure: str
    left_value: float
    right_value: float
    similarity: float
    delta: float


@dataclass(frozen=True, kw_only=True)
class DatasetProfile:
    compounds: int
    partition_counts: dict[str, int]
    target_kind: str
    # Exactly one of these two is populated, decided by `target_kind`. A numeric
    # target has no class balance and a binary one has no distribution worth
    # binning; carrying both as optional is what keeps a consumer from rendering
    # an empty histogram for a binary dataset.
    target_distribution: TargetDistribution | None = None
    class_balance: list[ClassBalance] = field(default_factory=list)
    # `None` when there was nothing to compare -- an empty train or test
    # partition -- never an all-zero histogram, which would read as "every test
    # compound is confirmed unlike anything trained on".
    similarity: SimilarityProfile | None = None
    # `None` for a sequence dataset. A Bemis-Murcko scaffold is a ring system, which an
    # amino-acid sequence does not have, so the honest profile omits the section rather
    # than reporting that every sequence is its own singleton family -- a true sentence
    # about a question that was never meaningful. Same reasoning for `similarity`,
    # `descriptors` and `activity_cliffs` below, all of which key off Tanimoto or RDKit.
    scaffolds: ScaffoldProfile | None = None
    descriptors: list[DescriptorProfile] = field(default_factory=list)
    best_descriptor: str | None = None
    activity_cliffs: list[ActivityCliff] = field(default_factory=list)
    # Non-null when the cliff scan ran on a subsample rather than every compound.
    # Reported rather than silently applied: a truncated search that presents as
    # exhaustive is the failure mode this field exists to prevent.
    cliffs_sampled_from: int | None = None


#: Bumped whenever `build_profile` changes what a number *means* -- a new
#: threshold, a different grouping, a corrected statistic. A cached profile
#: carrying an older version is recomputed rather than served.
#:
#: This is not paranoia, it is the bug this cache actually produced: the
#: cross-split scaffold count was corrected while a profile computed by the old
#: code sat in the blob store, and the page went on reporting the old, wrong
#: number with nothing to indicate it was stale. A shape change would have been
#: caught by `profile_from_dict` raising; a *value* change is invisible without
#: this.
PROFILE_VERSION = 2


def profile_to_dict(profile: DatasetProfile) -> dict[str, Any]:
    """Plain JSON for the cached blob.

    Hand-written rather than `dataclasses.asdict` so the stored shape is a
    decision rather than a consequence of the class layout: this blob is read
    back by a later deployment, and a field renamed in Python must not silently
    orphan every cached profile already written.
    """

    def histogram(value: Histogram) -> dict[str, Any]:
        return {"edges": value.edges, "counts": value.counts}

    def split_histogram(value: SplitHistogram) -> dict[str, Any]:
        return {"edges": value.edges, "train": value.train, "test": value.test}

    def summary(value: NumericSummary) -> dict[str, Any]:
        return {
            "minimum": value.minimum,
            "maximum": value.maximum,
            "mean": value.mean,
            "median": value.median,
            "std": value.std,
        }

    return {
        "version": PROFILE_VERSION,
        "compounds": profile.compounds,
        "partition_counts": profile.partition_counts,
        "target_kind": profile.target_kind,
        "target_distribution": (
            {
                "histogram": split_histogram(profile.target_distribution.histogram),
                "train": summary(profile.target_distribution.train),
                "test": summary(profile.target_distribution.test),
            }
            if profile.target_distribution
            else None
        ),
        "class_balance": [
            {"split": entry.split, "positive": entry.positive, "negative": entry.negative}
            for entry in profile.class_balance
        ],
        "similarity": (
            {
                "histogram": histogram(profile.similarity.histogram),
                "median": profile.similarity.median,
                "within_domain": profile.similarity.within_domain,
                "within_domain_threshold": profile.similarity.within_domain_threshold,
                "near_duplicates": profile.similarity.near_duplicates,
                "near_duplicate_threshold": profile.similarity.near_duplicate_threshold,
            }
            if profile.similarity
            else None
        ),
        "scaffolds": (
            {
                "unique_count": profile.scaffolds.unique_count,
                "singleton_count": profile.scaffolds.singleton_count,
                "largest_fraction": profile.scaffolds.largest_fraction,
                "cumulative_coverage": profile.scaffolds.cumulative_coverage,
                "top": [{"smiles": e.smiles, "count": e.count} for e in profile.scaffolds.top],
                "cross_split_scaffolds": profile.scaffolds.cross_split_scaffolds,
                "cross_split_compounds": profile.scaffolds.cross_split_compounds,
            }
            if profile.scaffolds
            else None
        ),
        "descriptors": [
            {
                "name": d.name,
                "histogram": split_histogram(d.histogram),
                "median": d.median,
                "target_correlation": d.target_correlation,
            }
            for d in profile.descriptors
        ],
        "best_descriptor": profile.best_descriptor,
        "activity_cliffs": [
            {
                "left_structure": c.left_structure,
                "right_structure": c.right_structure,
                "left_value": c.left_value,
                "right_value": c.right_value,
                "similarity": c.similarity,
                "delta": c.delta,
            }
            for c in profile.activity_cliffs
        ],
        "cliffs_sampled_from": profile.cliffs_sampled_from,
    }


def profile_from_dict(data: dict[str, Any]) -> DatasetProfile:
    def histogram(value: dict[str, Any]) -> Histogram:
        return Histogram(edges=value["edges"], counts=value["counts"])

    def split_histogram(value: dict[str, Any]) -> SplitHistogram:
        return SplitHistogram(edges=value["edges"], train=value["train"], test=value["test"])

    def summary(value: dict[str, Any]) -> NumericSummary:
        return NumericSummary(
            minimum=value["minimum"],
            maximum=value["maximum"],
            mean=value["mean"],
            median=value["median"],
            std=value["std"],
        )

    distribution = data.get("target_distribution")
    similarity = data.get("similarity")
    # `.get`, not `[...]`: a sequence dataset has no scaffold section at all.
    scaffolds = data.get("scaffolds")
    return DatasetProfile(
        compounds=data["compounds"],
        partition_counts=data["partition_counts"],
        target_kind=data["target_kind"],
        target_distribution=(
            TargetDistribution(
                histogram=split_histogram(distribution["histogram"]),
                train=summary(distribution["train"]),
                test=summary(distribution["test"]),
            )
            if distribution
            else None
        ),
        class_balance=[
            ClassBalance(
                split=entry["split"], positive=entry["positive"], negative=entry["negative"]
            )
            for entry in data.get("class_balance", [])
        ],
        similarity=(
            SimilarityProfile(
                histogram=histogram(similarity["histogram"]),
                median=similarity["median"],
                within_domain=similarity["within_domain"],
                within_domain_threshold=similarity["within_domain_threshold"],
                near_duplicates=similarity["near_duplicates"],
                near_duplicate_threshold=similarity["near_duplicate_threshold"],
            )
            if similarity
            else None
        ),
        scaffolds=(
            ScaffoldProfile(
                unique_count=scaffolds["unique_count"],
                singleton_count=scaffolds["singleton_count"],
                largest_fraction=scaffolds["largest_fraction"],
                cumulative_coverage=scaffolds["cumulative_coverage"],
                top=[
                    ScaffoldEntry(smiles=e["smiles"], count=e["count"]) for e in scaffolds["top"]
                ],
                cross_split_scaffolds=scaffolds["cross_split_scaffolds"],
                cross_split_compounds=scaffolds["cross_split_compounds"],
            )
            if scaffolds
            else None
        ),
        descriptors=[
            DescriptorProfile(
                name=d["name"],
                histogram=split_histogram(d["histogram"]),
                median=d["median"],
                target_correlation=d["target_correlation"],
            )
            for d in data.get("descriptors", [])
        ],
        best_descriptor=data.get("best_descriptor"),
        activity_cliffs=[
            ActivityCliff(
                left_structure=c["left_structure"],
                right_structure=c["right_structure"],
                left_value=c["left_value"],
                right_value=c["right_value"],
                similarity=c["similarity"],
                delta=c["delta"],
            )
            for c in data.get("activity_cliffs", [])
        ],
        cliffs_sampled_from=data.get("cliffs_sampled_from"),
    )
