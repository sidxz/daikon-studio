"""Whole-branch review follow-up (C1 staleness): `_final_labels` computes the
*actual* column/tag names an export would render, including the
`generation_method` column/tag `export_collection.py` always appends. This
checks that real output against `RESERVED_TARGET_COLUMNS`, rather than
trusting a hand-maintained mirror of what the module is believed to inject --
the exact gap that let an unrelated `_row_number` injection go unreserved.
"""

from __future__ import annotations

from daikonstudio.application.data.export_collection import ExportFormat, _final_labels
from daikonstudio.domain.data.target import RESERVED_TARGET_COLUMNS


def test_export_always_appends_a_reserved_provenance_column():
    for export_format in ExportFormat:
        # No readouts: isolates the column/tag this module appends
        # unconditionally, on every export, regardless of the Protocol.
        appended = _final_labels((), export_format)
        assert "generation_method" in appended
        assert set(appended) - {"smiles"} <= RESERVED_TARGET_COLUMNS
