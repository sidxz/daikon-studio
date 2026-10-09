"""`actual`, `predicted` and `baseline_predicted` must drop the same rows.

`predict` replays the whole test partition for every target (`PredictContext(frame=
test_rows, ...)`), so predictions always cover every test row -- including the ones
this target never measured. `actual` reads the target column, which is null there. Mask
one and not the others and the paired bootstrap from sub-project B resamples three
vectors that no longer describe the same compounds: it still returns an interval, and
the interval is about nothing.
"""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.application.execution.train_protocol import _labelled_vectors


def test_the_three_vectors_drop_the_same_rows() -> None:
    test_rows = pl.DataFrame({"tox": [1.0, None, 0.0, None, 1.0]})
    predicted = [0.9, 0.5, 0.2, 0.4, 0.8]
    baseline = [0.7, 0.6, 0.3, 0.1, 0.6]

    actual, model, base, indices = _labelled_vectors("tox", test_rows, predicted, baseline)

    assert actual == [1.0, 0.0, 1.0]
    assert model == [0.9, 0.2, 0.8]
    assert base == [0.7, 0.3, 0.6]
    # The positions these came from, so the run-level structures and chemistry can be
    # narrowed to the same compounds rather than paired by a position that has shifted.
    assert indices == [0, 2, 4]


def test_a_dense_column_keeps_every_row() -> None:
    test_rows = pl.DataFrame({"tox": [1.0, 0.0]})

    actual, model, base, indices = _labelled_vectors("tox", test_rows, [0.9, 0.1], [0.8, 0.2])

    assert (actual, model, base) == ([1.0, 0.0], [0.9, 0.1], [0.8, 0.2])
    # None, not `[0, 1]`: a dense target must leave every existing blob unchanged.
    assert indices is None


def test_no_baseline_stays_none() -> None:
    test_rows = pl.DataFrame({"tox": [1.0, None]})

    _, _, base, _ = _labelled_vectors("tox", test_rows, [0.9, 0.5], None)

    assert base is None


def test_a_fully_unmeasured_target_yields_empty_vectors() -> None:
    """A rare assay under a scaffold split can land every measurement in training.
    Empty is the honest answer; `build_scorecard` reports undefined metrics from it."""
    test_rows = pl.DataFrame({"tox": [None, None]})

    actual, model, base, indices = _labelled_vectors("tox", test_rows, [0.9, 0.1], [0.5, 0.5])

    assert (actual, model, base) == ([], [], [])
    assert indices == []


def test_a_prediction_vector_of_the_wrong_length_raises() -> None:
    """The bug this helper exists to prevent. A silently truncated vector would pair
    each compound's label with another compound's prediction and still produce a
    number, so the mismatch must stop the run rather than be absorbed."""
    test_rows = pl.DataFrame({"tox": [1.0, None, 0.0]})

    with pytest.raises(ValueError):
        _labelled_vectors("tox", test_rows, [0.9, 0.1], None)


def test_a_dense_column_also_rejects_a_wrong_length_vector() -> None:
    """The length check must not be reachable only on sparse data. A dense dataset with
    a mismatched prediction vector is the same bug and has always been silent."""
    test_rows = pl.DataFrame({"tox": [1.0, 0.0, 1.0]})

    with pytest.raises(ValueError):
        _labelled_vectors("tox", test_rows, [0.9, 0.1], None)
