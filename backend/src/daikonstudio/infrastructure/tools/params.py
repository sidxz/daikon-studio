"""The settings a step of each tool takes.

The planner fills these as data, and the coherence check rejects a step that is
missing one. Field descriptions are shown to the planner, so they say where a
value comes from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FetchPdbParams(BaseModel):
    pdb_id: str = Field(
        min_length=4,
        max_length=4,
        description="Four-character PDB id, as the scientist named it",
    )

    @field_validator("pdb_id")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class FetchUniprotParams(BaseModel):
    query: str | None = Field(
        default=None,
        description=(
            "UniProt accession, gene or protein name. Omit to take the accession "
            "from a structure fetched earlier in the plan"
        ),
    )


class FetchAlphafoldParams(BaseModel):
    accession: str = Field(
        min_length=6,
        description="UniProt accession of the protein to fetch the model for",
    )


class ActivityParams(BaseModel):
    activity_column: str | None = Field(
        default=None,
        description="Column holding a log potency, for ligand efficiency; omit if there is none",
    )


class ConfoundParams(BaseModel):
    activity_column: str = Field(
        description="Column holding the activity to test against the properties"
    )


class SubstructureParams(BaseModel):
    smarts: str = Field(
        min_length=1,
        description=(
            "The substructure to look for, as SMARTS. A plain SMILES is valid SMARTS, so "
            "a literal fragment such as c1ccccc1 works; use SMARTS only where the query "
            "needs wildcards or atom lists"
        ),
    )


class SimilarityParams(BaseModel):
    query_smiles: str = Field(
        min_length=1,
        description="The structure to compare every compound against, as SMILES",
    )
    cutoff: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description=(
            "Tanimoto at or above which a compound is reported as similar. Only counted, "
            "never used to drop rows. 0.7 on this fingerprint is the usual convention and "
            "is a convention, not a threshold with meaning"
        ),
    )


class ClusterParams(BaseModel):
    cutoff: float = Field(
        default=0.35,
        gt=0,
        le=1,
        description=(
            "Tanimoto *distance* below which two compounds join a cluster, so a smaller "
            "number makes tighter, more numerous clusters. 0.35 distance is the common "
            "default, equivalent to 0.65 similarity"
        ),
    )


class DiverseParams(BaseModel):
    n: int = Field(
        gt=0,
        description="How many compounds to pick. Fewer are returned if the table is smaller",
    )


class InteractionParams(BaseModel):
    poses_per_compound: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "How many poses per compound to analyse, best score first. One is the usual "
            "question; more is for asking whether the runner-up binds differently"
        ),
    )


class ValidityParams(BaseModel):
    keep_invalid: bool = Field(
        default=False,
        description=(
            "Keep the poses that fail, marked, instead of dropping them. Off, the output "
            "is the poses that passed and the failures are reported beside it"
        ),
    )


class DockParams(BaseModel):
    exhaustiveness: int = Field(
        default=8,
        ge=1,
        le=32,
        description=(
            "How hard to search per compound. 8 is the usual default; raising it costs "
            "time roughly in proportion and buys little on a well-placed box"
        ),
    )
    poses_per_compound: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many poses to keep for each compound, best score first",
    )
    max_compounds: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "A ceiling on how many compounds to dock, since each takes seconds to minutes. "
            "Compounds beyond it are left undocked and reported, never silently dropped"
        ),
    )
    seed: int = Field(
        default=42,
        description="Fixed so the same compounds in the same box give the same poses twice",
    )


class PrepareParams(BaseModel):
    ph: float = Field(
        default=7.4, ge=0, le=14, description="pH at which protonation states are assigned"
    )

    @field_validator("keep_ligands", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        """Accept a string, because the planner cannot send a list.

        Structured output carries parameters as name/value pairs, so every
        setting arrives as text. A float survives that; a list does not, and the
        plan is rejected with a validation error the scientist can do nothing
        about. "" and "[]" both mean strip everything, which is the answer that
        matters most here.
        """
        if not isinstance(value, str):
            return value
        cleaned = value.strip().strip("[]")
        return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]

    keep_ligands: list[str] | None = Field(
        default=None,
        description=(
            "Which heterogen components to keep in the receptor, by PDB chemical "
            "component id (for example NAD, HEM, ZN). Waters and crystallisation "
            "additives are always removed. Left unset, every non-additive ligand "
            "stays, which is right for a cofactor the protein works with and wrong "
            "for a bound inhibitor: keeping that fills the site you are looking "
            "for. Pass an empty list to strip them all."
        ),
    )


class AssayParams(BaseModel):
    """How the activity column is measured. The scientist's assay, never inferred."""

    activity_column: str = Field(
        description="Column holding the activity, as the scientist named it for this assay"
    )
    kind: Literal["log", "concentration", "linear"] = Field(
        description=(
            "What the column's numbers are, from the assay description or its header. "
            "log: already a log potency (pIC50, pEC50, pKi), so a bigger number is a "
            "stronger compound. concentration: a dose that does the job (IC50, EC50, "
            "MIC, Ki in nM or uM), so a smaller number is stronger and differences are "
            "taken on the log scale. linear: any other quantity (percent inhibition, "
            "a rate), compared as it stands"
        )
    )
    higher_is_better: bool = Field(
        description=(
            "Whether a larger value of the activity (after the kind's own conversion) "
            "is the better result, from what the assay measures. Getting this wrong "
            "reverses every conclusion the study draws, so it is asked, not guessed"
        )
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the column (nM, uM, %), if the scientist gave one; shown, not used",
    )

    @field_validator("higher_is_better", mode="before")
    @classmethod
    def _text_bool(cls, value: object) -> object:
        """Settings arrive as text (see `PrepareParams._split`), so accept yes/no."""
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "yes"):
                return True
            if text in ("false", "no"):
                return False
        return value


class CoreParams(BaseModel):
    core: str = Field(
        min_length=1, description="The core as the scientist typed it, SMILES or SMARTS"
    )


class CliffParams(BaseModel):
    min_delta: float = Field(
        gt=0,
        description=(
            "Smallest difference in activity, in comparable units, that makes a pair a "
            "cliff. 1.0 is tenfold for a potency or concentration; 0.6 is two doubling "
            "dilutions, the usual MIC floor; for a linear quantity it is the column's "
            "own units. Required because the floor belongs to the assay, not to the tool"
        ),
    )
