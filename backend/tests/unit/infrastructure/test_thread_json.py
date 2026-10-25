from datetime import UTC, datetime

from daikonstudio.domain.studies.study import Thread, Turn
from daikonstudio.infrastructure.persistence.sqlalchemy.studies.repository import (
    _thread_from_json,
    _thread_to_json,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_a_turn_stored_before_wanted_existed_reads_as_empty() -> None:
    old = {"role": "agent", "say": "s", "at": "2026-01-01T00:00:00+00:00"}
    turn = _thread_from_json({"summary": "", "turns": [old]}).turns[0]
    assert turn.wanted == () and turn.target_kinds == ()


def test_wanted_and_target_kinds_round_trip() -> None:
    t = Thread(
        turns=[
            Turn(
                role="agent",
                kind="asking",
                say="s",
                wanted=("find_cores",),
                target_kinds=("core",),
                at=_T0,
            )
        ]
    )
    back = _thread_from_json(_thread_to_json(t)).turns[0]
    assert back.wanted == ("find_cores",) and back.target_kinds == ("core",)
