"""The settings a step of each tool takes.

The planner fills these as data, and the coherence check rejects a step that is
missing one. Field descriptions are shown to the planner, so they say where a
value comes from.
"""

from __future__ import annotations

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
