"""The workbook's class column and its stated cutoff, which the API tests (a
regression Protocol) never reach."""

import polars as pl

from daikonstudio.application.execution.export_results import _cutoff, _meanings, _shown
from daikonstudio.domain.catalog.readout import Readout, ReadoutType


def _class(threshold: float | None) -> Readout:
    return Readout(
        name="p_np",
        type=ReadoutType.CLASS,
        unit=None,
        direction=None,
        description="Predicted p_np class",
        threshold=threshold,
    )


def test_a_stored_class_is_named_and_a_missing_one_stays_empty():
    frame = pl.DataFrame({"p_np": [1.0, 0.0, None]})
    shown = frame.select(_shown("p_np", pl.Float64(), _class(0.584)))["p_np"].to_list()
    assert shown == ["Active", "Inactive", None]


def test_the_about_sheet_states_the_protocols_own_cutoff():
    class Protocol:
        readouts = (_class(0.584),)

    [(_, text)] = _meanings(Protocol(), {"p_np": "p_np"})  # type: ignore[arg-type]
    assert "at least 0.584" in text
    Protocol.readouts = (_class(None),)
    [(_, text)] = _meanings(Protocol(), {"p_np": "p_np"})  # type: ignore[arg-type]
    assert "at least 0.5," in text


def test_a_cutoff_reads_as_the_scorecard_shows_it():
    assert _cutoff(0.4124307930469513) == "0.412"
    assert _cutoff(0.5) == "0.5"
    # Rounded to three digits this is 1, which would say nothing is ever active.
    assert _cutoff(0.99997) == "0.99997"
