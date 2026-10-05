import uuid

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.interface.routes.runs import RunResponse


def _run(params):
    return Run(
        kind=RunKind.PREDICTION,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="k",
        params=params,
        protocol_id=uuid.uuid4(),
    )


def test_a_run_answers_with_who_started_it_and_where_its_compounds_came_from():
    source = {
        "app": "chemcellar",
        "run_id": str(uuid.uuid4()),
        "protocol_id": str(uuid.uuid4()),
        "protocol_name": "NadD-Sumo dose response",
        "run_date": "2026-06-05",
        "compounds_without_structure": 2,
    }
    run = _run({"upload_ref": "x", "source": source})
    response = RunResponse.from_domain(run)
    assert response.requested_by == run.requested_by
    assert response.source is not None
    assert (response.source.protocol_name, str(response.source.run_date)) == (
        "NadD-Sumo dose response",
        "2026-06-05",
    )


def test_a_run_from_a_file_has_no_source():
    assert RunResponse.from_domain(_run({"upload_ref": "x"})).source is None
