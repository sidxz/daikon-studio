"""The Tool contract's declarative half.

A tool is one step a scientist already performs by hand: standardize structures,
propose a core, decompose R groups, find the pairs that differ in one place. The
manifest is what a planner reads to decide which tools answer a question and in
what order, and what the catalogue page renders. Like `engines/manifest.py` it is
deliberately free of infrastructure imports, so the envelope serializes to HTTP
unchanged and so a tool that later runs out of process needs no second shape.

Two things are deliberately *not* here.

No runtime parameter spec. `params` below is only the schema a planner fills in
and the coherence check validates; what a setting does at run time is not decided
here.

No "where the table came from". A tool consumes and produces slots (below) and
says nothing about whether a compound table arrived as a Dataset, an upload, or
another tool's output. That boundary is still being found, and a manifest that
guessed at it would have to be rewritten once it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel


class SlotKind(StrEnum):
    """What flows between tools.

    The planner chains tools by matching an output slot to an input slot, so this
    enum is the vocabulary a plan is written in. Keep it small: a kind earns its
    place when two tools genuinely pass the same thing between them, not when one
    tool happens to return something with a name.
    """

    # Rows of compounds: a structure column, an identifier, and whatever else came
    # with them. The thing almost every tool starts from.
    COMPOUND_TABLE = "compound_table"
    # One scaffold, as SMARTS. The single most consequential value in an SAR study:
    # everything downstream is expressed relative to it.
    CORE = "core"
    # Ranked cores for a human to choose between, each with what it covers and what
    # it gave up. Never collapsed to one by a machine -- see `judgment` below.
    CORE_CANDIDATES = "core_candidates"
    # Groups of compounds that share chemistry, with the core each group would give.
    SERIES = "series"
    # Compounds decomposed against a core: one row per compound, one column per
    # attachment point. The SAR table itself.
    RGROUP_TABLE = "rgroup_table"
    # Pairs of compounds identical but for one position, with the change and the
    # difference in activity.
    PAIRS = "pairs"
    # A per-transform summary over many pairs: how much a change is worth, over how
    # many pairs, and how wide the interval is.
    EFFECTS = "effects"
    # Physicochemical values per compound.
    PROPERTY_TABLE = "property_table"
    # One statistic with its interval, for a question of the form "is activity
    # really tracking this?"
    CORRELATION = "correlation"
    # Flagged substructures per compound or per R group.
    ALERTS = "alerts"
    # A fitted model's terms and how well it holds.
    MODEL_FIT = "model_fit"
    # Combinations the series has never made.
    GAPS = "gaps"
    # References with links that resolve. Citations, never claims.
    CITATIONS = "citations"
    # One protein structure, repaired and protonated. Brought by the scientist as
    # a file, or fetched by id from a public archive; either way, prepared before
    # anything else touches it.
    STRUCTURE = "structure"
    # Curated residue-level facts about a protein -- binding, active and metal
    # sites, known variants -- numbered on its canonical sequence. What lets a
    # detected pocket be told apart from the site the literature already knows.
    SITE_ANNOTATIONS = "site_annotations"
    # Detected binding sites, ranked, for a person to choose between.
    POCKET_CANDIDATES = "pocket_candidates"
    # The site the work is aimed at. Approved, never computed -- the same seam as
    # a core, for the same reason.
    POCKET = "pocket"
    # Predicted absorption, distribution, metabolism, excretion and toxicity
    # values per compound.
    ADMET_TABLE = "admet_table"


# The lane a tool runs in when it asks for nothing special. A lane is a
# requirement ("this needs the torch stack", "this needs native binaries"), never
# a machine -- the same meaning `EngineManifest.lane` already carries, and served
# by the same runner fleet.
DEFAULT_LANE = "default"


class ToolArea(StrEnum):
    """Which kind of work a tool belongs to.

    Only a grouping, never a restriction: a plan may mix areas freely, and a
    structural alert is as useful in a hit list as in a series. It exists because
    a flat catalogue stops being readable somewhere around twenty entries, and
    because a chemist arriving for one kind of work should not have to read past
    the rest.

    Required, with no default. A new tool has to say where it belongs, which is a
    question worth one line at the time of writing and a mess to answer later for
    forty entries at once.
    """

    # Preparation and per-compound facts. Useful in any analysis, not tied to one.
    DATA = "data"
    # Understanding a congeneric series: cores, R groups, pairs, cliffs.
    SAR = "sar"
    # Anything that needs a protein structure. Nothing yet; the second case study
    # brings pocket detection and docking.
    STRUCTURE = "structure"
    # What is already known elsewhere: public activity data, the literature.
    CONTEXT = "context"


class ToolSurface(StrEnum):
    """How much of a person a tool's result needs when it finishes.

    This is the cost driver for the catalogue, not a cosmetic hint: an EDITOR tool
    owes a real interface, a VIEW tool owes a readable rendering, and a NONE tool
    owes a count. Declaring it here keeps that cost visible before a tool is added
    rather than after.
    """

    # A table and a count. No judgement is available at this step, so there is
    # nothing for a person to decide.
    NONE = "none"
    # A plot or table good enough to judge from, read only.
    VIEW = "view"
    # Direct manipulation. The scientist does not tune a parameter and re-run, they
    # supply the answer -- which is a different kind of value, and the reason the
    # study has to record who said so.
    EDITOR = "editor"


@dataclass(frozen=True, kw_only=True)
class Slot:
    key: str
    kind: SlotKind
    label: str
    # A slot a tool can do without. An input that is absent changes what the tool
    # reports rather than failing it: `property_confound` without a property table
    # computes the properties it needs, `find_cores` without a series works on the
    # whole set.
    optional: bool = False


@dataclass(frozen=True, kw_only=True)
class ToolManifest:
    id: str
    version: str
    name: str
    # One or two sentences in a chemist's words: what it does, and the thing it
    # gets right that a naive version gets wrong. This is read by a planner
    # choosing between tools *and* by a scientist reading the catalogue, and those
    # are the same need -- a description vague enough to mislead the planner is
    # also too vague for the page.
    description: str
    area: ToolArea
    inputs: tuple[Slot, ...]
    outputs: tuple[Slot, ...]
    # What actually does the work, named so a study can cite it and an operator can
    # tell what a version bump moved.
    library: str
    # Returns ranked candidates for a person to choose between, rather than an
    # answer. The planner may not collapse these itself. The case that forces the
    # rule: on a benzamide series the top-ranked core reaches full coverage only by
    # dropping the amide carbonyl so a thioamide fits, and a chemist would take the
    # narrower core instead. Rank one is not the answer, it is the first option.
    judgment: bool = False
    # What a chosen candidate becomes, for a tool that returns candidates. This is
    # how a value that no tool produces comes to exist: nothing in the catalogue
    # emits a CORE, because a core is not computed, it is approved. `find_cores`
    # emits candidates and a chemist's choice resolves one into the core the rest
    # of the study is expressed against.
    #
    # Worth keeping explicit rather than inferring it: it is the seam where a
    # computed value becomes an asserted one, and a study has to record who
    # asserted it. A planner that chained straight from candidates to the next
    # tool would have silently made that choice on the chemist's behalf.
    resolves_to: SlotKind | None = None
    # The tools this one runs, in order. A composite: one entry in the catalogue
    # that stands for a sequence everybody runs together anyway.
    #
    # It exists for the planner, not for the runtime. Picking which tools to use
    # is where a planner goes wrong, so offering "build the SAR table" instead of
    # four separate steps removes four chances to pick a wrong order. A plan
    # expands it before anything executes, so the record keeps every individual
    # step, its gate and its provenance -- a composite is a named plan fragment,
    # never a second way of sequencing work.
    #
    # One level only. Composites of composites would be a second plan language,
    # and `test_composites_do_not_nest` keeps it that way.
    expands_to: tuple[str, ...] = ()
    # Runs whenever its inputs are present, and is not the planner's to choose.
    #
    # Some steps do not depend on anything in the data or on what the scientist
    # wants. You standardize structures and merge duplicates because the same
    # compound drawn twice is two compounds, and that is true of every table that
    # ever arrives. Leaving such a step to a model means it can be forgotten, and
    # a plan missing it is silently wrong rather than visibly broken -- nothing
    # downstream *requires* clean input in slot terms, so no check would catch it.
    #
    # The plan prepends these instead, and the model is told they have happened so
    # it neither repeats them nor asks about them.
    automatic: bool = False
    surface: ToolSurface = ToolSurface.NONE
    # Runs outside this process, so it can fail or change version independently and
    # a study has to pin what it was given.
    external: bool = False
    # Which runner lane can serve this tool. The grouping axis is the dependency
    # set, not the tool: everything that needs only RDKit shares the API's own
    # environment, everything that needs the torch stack shares the lane that
    # already has it, and native binaries share one image because that is where
    # the licence and the apt packages live. Three environments, not nineteen.
    lane: str = DEFAULT_LANE
    # The settings a step of this tool takes, as a pydantic model. The planner
    # fills them as data and `incoherences` validates them; the model lives in
    # infrastructure and arrives here only as a type, so application stays clean.
    params: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if self.judgment and self.surface is ToolSurface.NONE:
            raise ValueError(
                f"Tool {self.id!r} returns candidates for a person to choose between "
                f"but offers them no surface to choose on."
            )
        if self.expands_to and len(set(self.expands_to)) != len(self.expands_to):
            raise ValueError(f"Tool {self.id!r} lists the same step twice.")
        if self.resolves_to is not None and not self.judgment:
            raise ValueError(
                f"Tool {self.id!r} says what a chosen candidate resolves to, but does "
                f"not return candidates for anyone to choose between."
            )
