"""A training run's finished epochs: a runner posts them, the run page reads them.

What the live charts on a training run's page are drawn from."""

from __future__ import annotations

from datetime import UTC, datetime

from tests.helpers.runner_fixtures import claim, register_runner, seed_run

from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunKind


def _point(epoch: int, **overrides: object) -> dict[str, object]:
    point: dict[str, object] = {
        "epoch": epoch,
        "epochs": 30,
        "train_loss": 0.7 - epoch / 100,
        "val_loss": 0.69 - epoch / 200,
        "scores": {"auroc": 0.70 + epoch / 100, "auprc": 0.4, "mcc": 0.2},
        "device": "cuda:0",
        "target": "aggregator",
        "fit": "model",
        "at": datetime.now(UTC).isoformat(),
    }
    point.update(overrides)
    return point


async def _claimed_training_run(app, anonymous_client, workspace_id, requested_by=None):
    run = await seed_run(app, workspace_id, kind=RunKind.TRAINING, requested_by=requested_by)
    _, headers = await register_runner(app, ["default"])
    await claim(anonymous_client, headers)
    return run, headers


async def test_the_page_reads_back_what_the_runner_posted(
    app, anonymous_client, client, workspace_id, client_user_id
):
    run, headers = await _claimed_training_run(app, anonymous_client, workspace_id, client_user_id)

    posted = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs",
        headers=headers,
        json={"points": [_point(1), _point(2)]},
    )
    assert posted.status_code == 204, posted.text

    epochs = (await client.get(f"/api/v1/runs/{run.id}/epochs")).json()

    assert [point["epoch"] for point in epochs] == [1, 2]
    assert epochs[1]["scores"] == {"auroc": 0.72, "auprc": 0.4, "mcc": 0.2}
    assert epochs[0]["target"] == "aggregator"
    assert epochs[0]["device"] == "cuda:0"


async def test_another_workspace_cannot_read_a_runs_epochs(
    app, anonymous_client, other_workspace_client, workspace_id
):
    run, headers = await _claimed_training_run(app, anonymous_client, workspace_id)
    await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs", headers=headers, json={"points": [_point(1)]}
    )

    assert (await other_workspace_client.get(f"/api/v1/runs/{run.id}/epochs")).status_code == 404


async def test_only_the_claimant_may_post_epochs(app, anonymous_client, workspace_id):
    run, _ = await _claimed_training_run(app, anonymous_client, workspace_id)
    _, stranger = await register_runner(app, ["default"])

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs", headers=stranger, json={"points": [_point(1)]}
    )

    assert response.status_code == 403


async def test_a_point_out_of_bounds_is_rejected(app, anonymous_client, workspace_id):
    run, headers = await _claimed_training_run(app, anonymous_client, workspace_id)

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs",
        headers=headers,
        json={"points": [_point(0)]},  # epochs are 1-based
    )

    assert response.status_code == 422


async def test_a_redelivered_run_is_charted_from_its_latest_attempt_alone(
    app, anonymous_client, client, workspace_id, client_user_id
):
    """A runner died mid-fit and the run was claimed again: the second attempt starts
    over at epoch 1, and drawing both would zigzag the curves."""
    run, headers = await _claimed_training_run(app, anonymous_client, workspace_id, client_user_id)
    await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs",
        headers=headers,
        json={"points": [_point(1), _point(2), _point(3)]},
    )
    runs = app.state.container[RunRepository]
    # The second claim, as the queue records it.
    async with runs._sessions() as session:
        from sqlalchemy import text

        await session.execute(
            text("UPDATE runs SET attempts = attempts + 1 WHERE id = :id"), {"id": run.id}
        )
        await session.commit()
    await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/epochs", headers=headers, json={"points": [_point(1)]}
    )

    epochs = (await client.get(f"/api/v1/runs/{run.id}/epochs")).json()

    assert [point["epoch"] for point in epochs] == [1]
