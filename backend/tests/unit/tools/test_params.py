from __future__ import annotations

import pytest

from daikonstudio.infrastructure.tools.params import AssayParams


@pytest.mark.parametrize(
    ("given", "want"), [("true", True), ("No", False), ("yes", True), ("false", False)]
)
def test_higher_is_better_arrives_as_text(given: str, want: bool) -> None:
    # A header the hint table does not recognise, so this tests the text coercion alone
    # and not the direction check that MIC would now (rightly) fail.
    p = AssayParams(activity_column="readout", kind="concentration", higher_is_better=given)  # type: ignore[arg-type]
    assert p.higher_is_better is want
