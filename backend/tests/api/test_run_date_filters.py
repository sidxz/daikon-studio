import uuid
from datetime import datetime

import pytest

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)


async def test_run_dates_filter_before_pagination_and_include_start_exclude_end(
    client, session_factory, workspace_id, client_user_id
):
    repository = SqlAlchemyRunRepository(session_factory)
    stamps = [
        "2026-10-05T04:59:59+00:00",  # before local October 5
        "2026-10-05T05:00:00+00:00",  # inclusive start
        "2026-10-06T04:59:59+00:00",  # final second of local October 5
        "2026-10-06T05:00:00+00:00",  # exclusive next midnight
    ]
    runs = []
    for stamp in stamps:
        run = Run(
            kind=RunKind.PREDICTION,
            workspace_id=workspace_id,
            requested_by=client_user_id,
            cache_key=str(uuid.uuid4()),
            created_at=datetime.fromisoformat(stamp),
        )
        await repository.add(run)
        runs.append(run)
    foreign = Run(
        kind=RunKind.PREDICTION,
        workspace_id=uuid.uuid4(),
        requested_by=client_user_id,
        cache_key=str(uuid.uuid4()),
        created_at=datetime.fromisoformat(stamps[2]),
    )
    await repository.add(foreign)
    params = {
        "kind": "prediction",
        "mine": "true",
        "status": "pending",
        "created_from": "2026-10-05T00:00:00-05:00",
        "created_before": "2026-10-06T00:00:00-05:00",
        "limit": 1,
    }
    first = await client.get("/api/v1/runs", params=params)
    assert first.status_code == 200, first.text
    assert [item["id"] for item in first.json()["items"]] == [str(runs[2].id)]
    assert first.json()["next_cursor"]
    second = await client.get(
        "/api/v1/runs", params={**params, "cursor": first.json()["next_cursor"]}
    )
    assert second.status_code == 200, second.text
    assert [item["id"] for item in second.json()["items"]] == [str(runs[1].id)]
    assert second.json()["next_cursor"] is None
    for boundary, expected in [
        ({"created_from": params["created_from"]}, runs[1:]),
        ({"created_before": params["created_before"]}, runs[:3]),
    ]:
        response = await client.get("/api/v1/runs", params={"kind": "prediction", **boundary})
        assert response.status_code == 200, response.text
        assert {item["id"] for item in response.json()["items"]} == {
            str(run.id) for run in expected
        }


@pytest.mark.parametrize(
    "params",
    [
        {"created_from": "2026-10-05T00:00:00"},
        {"created_before": "invalid"},
        {"created_from": "2026-10-06T00:00:00Z", "created_before": "2026-10-05T00:00:00Z"},
        {"created_from": "2026-10-05T00:00:00Z", "created_before": "2026-10-05T00:00:00Z"},
    ],
)
async def test_run_dates_reject_invalid_or_reversed_bounds(client, params):
    response = await client.get("/api/v1/runs", params=params)
    assert response.status_code == 422, response.text
