"""The params round trip, including runs written before the baseline was choosable.

`run.params` is write-once and never migrated, so `from_params` must keep
reading rows created by an older deploy. That is the regression these defaults
exist to prevent, not a hypothetical.
"""

from __future__ import annotations

import uuid

from daikonstudio.application.execution.train_protocol import TrainProtocolCommand


def test_params_round_trip_carries_the_baseline_pair() -> None:
    command = TrainProtocolCommand(
        name="BBBP — D-MPNN",
        dataset_id=uuid.uuid4(),
        engine_id="chemprop-dmpnn",
        conditions={"pretrained": "CheMeleon"},
        baseline_engine_id="chemprop-dmpnn",
        baseline_conditions={"pretrained": "none"},
    )
    restored = TrainProtocolCommand.from_params(command.to_params())
    assert restored == command


def test_from_params_reads_a_run_written_before_the_baseline_was_choosable() -> None:
    legacy = {
        "name": "ESOL — RF",
        "dataset_id": str(uuid.uuid4()),
        "engine_id": "ecfp4-randomforest",
        "conditions": {"n_estimators": 500},
    }
    restored = TrainProtocolCommand.from_params(legacy)
    assert restored.baseline_engine_id is None
    assert restored.baseline_conditions == {}
