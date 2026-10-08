"""Each training leg owns a slice of the progress bar, so a long fit has somewhere to
move instead of the bar sitting at one number. The slices are arithmetic, and arithmetic
that overlaps or runs backwards shows up as a progress bar that jumps about -- which no
integration test notices and every user does.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from daikonstudio.application.execution.train_protocol import (
    _RANDOM_SPLIT_SPAN,
    _leg_spans,
)


def test_no_draws_leaves_the_existing_span_untouched() -> None:
    """No existing run's progress bar may move because draws became possible."""
    gap, draws = _leg_spans(0)
    assert gap == _RANDOM_SPLIT_SPAN
    assert draws == []


def test_a_negative_count_is_treated_as_none() -> None:
    assert _leg_spans(-1) == (_RANDOM_SPLIT_SPAN, [])


@pytest.mark.parametrize("replicates", [1, 2, 3, 5, 10])
def test_draw_spans_tile_the_band_in_order(replicates: int) -> None:
    gap, draws = _leg_spans(replicates)

    assert len(draws) == replicates
    # The gap still starts where it always did; the draws take the rest of its band.
    assert gap[0] == _RANDOM_SPLIT_SPAN[0]
    assert draws[0][0] == pytest.approx(gap[1])
    assert draws[-1][1] == pytest.approx(_RANDOM_SPLIT_SPAN[1])
    for (start, end), (next_start, _) in pairwise(draws):
        assert start < end
        assert end == pytest.approx(next_start)
    # Monotonic, and never past the band it was given.
    for start, end in draws:
        assert _RANDOM_SPLIT_SPAN[0] <= start < end <= _RANDOM_SPLIT_SPAN[1]


def test_the_gap_keeps_room_to_report_when_draws_follow() -> None:
    """Squeezed, not erased: the gap leg still writes progress, so its slice cannot be
    empty or its own report would move the bar backwards into the first draw."""
    gap, _draws = _leg_spans(3)
    assert gap[0] < gap[1]
