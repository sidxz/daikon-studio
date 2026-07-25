"""Shared scientific-provenance value objects (who observed a fact, cited where).

Distinct from the persistence-layer ``ProvenanceMixin`` (import lineage:
source/release/checksum). A typed record may carry both.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from daikonstudio.domain.shared.errors import ValidationError


class ProvenanceSourceType(StrEnum):
    PUBLISHED = "published"
    PREPRINT = "preprint"
    PRIVATE_COMM = "private_comm"
    INTERNAL = "internal"
    PATENT = "patent"


class GenerationMethod(StrEnum):
    """How a provenance-stamped value was produced — orthogonal to source_type.

    source_type answers *publication status* (published/preprint/…); this answers
    *how the value came to be*. An AI can extract a value from a published paper,
    so both axes must coexist.
    """

    MANUAL = "manual"  # a human typed / curated it (the default)
    IMPORTED = "imported"  # loaded verbatim from an external DB / dataset
    AI_EXTRACTED = "ai_extracted"  # an AI pulled a stated value from a source
    AI_PREDICTED = "ai_predicted"  # an AI inferred a value not directly stated
    COMPUTED = "computed"  # a deterministic pipeline derived it


@dataclass(frozen=True, kw_only=True)
class Citation:
    """A literature reference. At least one identifier must be present."""

    pmid: str | None = None
    doi: str | None = None
    url: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not any((self.pmid, self.doi, self.url, self.label)):
            raise ValidationError("Citation requires at least one of pmid/doi/url/label")


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """Where a target-biology fact came from — the shared provenance envelope."""

    source_type: ProvenanceSourceType
    generation_method: GenerationMethod = GenerationMethod.MANUAL
    citations: tuple[Citation, ...] = ()
    contributor_researcher: str | None = None
    contributor_organization_id: uuid.UUID | None = None
    observed_on: date | None = None
    note: str | None = None
