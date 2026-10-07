import uuid
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daikonstudio.application.catalog.get_scorecard_tolerance import (
    GetScorecardTolerance,
    GetScorecardToleranceQuery,
    _count,
)
from daikonstudio.application.execution.build_scorecard import HeldOutChemistry
from daikonstudio.application.execution.train_protocol import scorecard_chemistry_key
from daikonstudio.domain.shared.errors import NotFoundError, ValidationError
from daikonstudio.interface.dependencies._core import get_auth
from daikonstudio.interface.error_handlers import register_error_handlers
from daikonstudio.interface.routes.protocols import router
from tests.fakes.auth import FakeAuth
from tests.fakes.protocol_access import FakeProtocolAccess
from tests.unit.catalog.test_get_scorecard import (
    _FakeProtocol,
    _FakeProtocols,
    _FakeStore,
    _inputs,
    _target,
)


@pytest.fixture
def setup():
    auth = FakeAuth(workspace_role="admin")
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    inputs = replace(
        _inputs(protocol.id),
        targets=[
            _target("first", actual=[0.0] * 4002, predicted=[0.0, 2.0] * 2001),
            _target("second", actual=[1.0, 2.0, 3.0], predicted=[1.0, 1.5, 2.0]),
            _target("binary", prediction_kind="probability"),
        ],
    )
    service = GetScorecardTolerance(
        _FakeProtocols(protocol), _FakeStore(inputs.to_json()), FakeProtocolAccess()
    )
    return auth, protocol, service


async def test_tolerance_uses_the_selected_target_full_population_and_inclusive_boundary(setup):
    auth, protocol, service = setup
    for target, tolerance, within, total in [
        ("first", 0, 2001, 4002),
        ("first", 2, 4002, 4002),
        ("second", 0.5, 2, 3),
    ]:
        result = (
            await service(
                GetScorecardToleranceQuery(
                    protocol_id=protocol.id, target=target, tolerance=tolerance
                ),
                auth,
            )
        ).unwrap()
        assert (result.within_count, result.test_count) == (within, total)


async def test_tolerance_obeys_visibility_and_rejects_unknown_or_binary_targets(setup):
    auth, protocol, service = setup
    missing = await service(
        GetScorecardToleranceQuery(protocol_id=uuid.uuid4(), target="first", tolerance=1), auth
    )
    assert isinstance(missing.failure(), NotFoundError)
    hidden = await service(
        GetScorecardToleranceQuery(protocol_id=protocol.id, target="first", tolerance=1),
        replace(auth, workspace_role="editor"),
    )
    assert isinstance(hidden.failure(), NotFoundError)
    for target, error in [("missing", NotFoundError), ("binary", ValidationError)]:
        result = await service(
            GetScorecardToleranceQuery(protocol_id=protocol.id, target=target, tolerance=1), auth
        )
        assert isinstance(result.failure(), error)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
async def test_invalid_tolerances_are_rejected_by_the_use_case(setup, tolerance):
    auth, protocol, service = setup
    result = await service(
        GetScorecardToleranceQuery(protocol_id=protocol.id, target="first", tolerance=tolerance),
        auth,
    )
    assert isinstance(result.failure(), ValidationError)


def test_http_tolerance_contract_and_validation(setup):
    auth, protocol, service = setup
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_auth] = lambda: auth
    app.state.container = {GetScorecardTolerance: service}
    register_error_handlers(app)
    with TestClient(app) as client:
        url = f"/api/v1/protocols/{protocol.id}/scorecard/tolerance"
        response = client.get(url, params={"target": "second", "tolerance": 0.5})
        assert response.status_code == 200
        assert response.json() == {
            "target": "second",
            "tolerance": 0.5,
            "within_count": 2,
            "test_count": 3,
            "by_similarity": [],
        }
        for tolerance in ["-1", "nan", "inf", "nonsense"]:
            assert (
                client.get(url, params={"target": "second", "tolerance": tolerance}).status_code
                == 422
            )


def test_tolerance_similarity_counts_are_exact_and_keep_the_same_groups():
    protocol_id = uuid.uuid4()
    inputs = replace(
        _inputs(protocol_id),
        targets=[
            _target("y", actual=[0.0] * 4002, predicted=[0.0, 2.0] * 2001),
        ],
    )
    similarities = [i / 4002 for i in range(4002)]
    narrow = _count(
        inputs.to_json(),
        "y",
        GetScorecardToleranceQuery(
            protocol_id=protocol_id,
            target="y",
            tolerance=0,
        ),
        similarities,
    ).unwrap()
    wide = _count(
        inputs.to_json(),
        "y",
        GetScorecardToleranceQuery(
            protocol_id=protocol_id,
            target="y",
            tolerance=2,
        ),
        similarities,
    ).unwrap()
    assert sum(b.count for b in narrow.by_similarity) == 4002
    assert sum(b.within_count for b in narrow.by_similarity) == 2001
    assert sum(b.within_count for b in wide.by_similarity) == 4002
    assert [(b.lower, b.upper, b.count) for b in narrow.by_similarity] == [
        (b.lower, b.upper, b.count) for b in wide.by_similarity
    ]


async def test_tolerance_reads_saved_similarity_data():
    auth = FakeAuth(workspace_role="admin")
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    inputs = replace(
        _inputs(protocol.id),
        targets=[
            _target("y", actual=[0.0] * 16, predicted=[0.0, 3.0] * 8),
        ],
    )
    store = _FakeStore(inputs.to_json())
    store.put_bytes(
        scorecard_chemistry_key(auth.workspace_id, protocol.id),
        HeldOutChemistry(similarities=[i / 16 for i in range(16)], scaffolds=[""] * 16).to_json(),
    )
    service = GetScorecardTolerance(_FakeProtocols(protocol), store, FakeProtocolAccess())
    result = (
        await service(
            GetScorecardToleranceQuery(protocol_id=protocol.id, target="y", tolerance=0), auth
        )
    ).unwrap()
    assert len(result.by_similarity) == 8
    assert all(b.within_count == 1 and b.count == 2 for b in result.by_similarity)
