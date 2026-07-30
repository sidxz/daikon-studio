# Phase 2 UX Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make run results sortable and filterable server-side, fuse the Scorecard into a single verdict strip, enrich the predict flow with a real preview, and render structures theme-aware — then verify the two paths no browser has seen.

**Architecture:** The backend's `GetPredictionResults` use case gains typed sort/filter fields applied as Polars operations on the already-in-memory results frame, plus a `row_id` minted before reordering so client-side row identity survives it. The frontend translates AG Grid's `sortModel`/`filterModel` into those parameters through one pure, tested function, and the display layers (Scorecard, predict wizard, structure thumbnail) are reshaped without touching their underlying logic.

**Tech Stack:** Python 3.12 / FastAPI / Polars / pytest (backend, `uv`); Next.js 15 / React / TypeScript / shadcn-ui / AG Grid Community / RDKit.js / TanStack Query / vitest / biome (frontend, `pnpm`).

**Spec:** `docs/superpowers/specs/2026-07-29-daikon-studio-phase-2-ux-design.md`

## Global Constraints

- **Branch:** `feat/frontend`. No git remote exists; commits are local only.
- **Never edit identity-service** (code, config, grants, deploys) without explicit in-the-moment consent. Not needed by any task here.
- **Every number in the UI renders through `<ReadoutValue>`** — the single choke point keeping unit and direction attached.
- **Absence renders as absence.** Null uncertainty/applicability/noise-floor render an em dash or are omitted entirely — never zero, never an empty box.
- **No UUIDs as input or display.** Named pickers; runs identified by protocol and date.
- **Count placement follows semantics:** operate-on counts inside the button (`Score 120 compounds`); forecast counts beside it.
- **orval generates types only; hooks are hand-written** against `customInstance`. Alias a generated DTO under a domain name; never redeclare its shape.
- **Feature structure:** `src/features/<name>/{components,hooks,lib,types}` with a hand-curated `index.ts` barrel — never `export *`.
- **Verify lint and types by exit code, never by eyeballing output.** Frontend: `cd frontend && pnpm lint` and `pnpm exec tsc --noEmit`. Backend: `cd backend && uv run pytest`.
- **A stale uvicorn serves a schema that no longer matches the code you are reading.** After any backend change, restart it (`make dev-be`) before testing the frontend against it.
- Regenerate the API contract with `make generate-api` (writes `frontend/openapi.json`, then runs `pnpm generate:api`).

---

## File Structure

**Backend**

| File | Responsibility |
|---|---|
| `backend/src/daikonstudio/application/execution/result_view.py` (new) | `SortSpec`, `RangeFilter`, and `apply_result_view` — the pure filter/sort/`row_id` logic |
| `backend/src/daikonstudio/application/execution/predict_with_protocol.py` | Extended `GetPredictionResultsQuery`; calls `apply_result_view`; `PredictionRow.row_id` |
| `backend/src/daikonstudio/interface/routes/runs.py` | Wire-format parsing of `sort_by`/`sort_dir`/`filters`; `row_id` on `PredictionResponse` |
| `backend/tests/api/test_runs.py` | HTTP-level tests for the new parameters |
| `backend/tests/unit/execution/test_result_view.py` (new) | Ordering, null placement, and `row_id` stability |

**Frontend**

| File | Responsibility |
|---|---|
| `frontend/src/features/runs/lib/result-query.ts` (new) | Pure translation of AG Grid sort/filter models + the in-domain switch into API params |
| `frontend/src/features/runs/lib/result-query.test.ts` (new) | Its tests |
| `frontend/src/features/runs/hooks/use-runs.ts` | `fetchResultBlock` gains sort/filter params; `__rowId` comes from the API |
| `frontend/src/features/runs/components/triage-grid.tsx` | Sortable/filterable columns, in-domain `Switch` |
| `frontend/src/features/runs/components/run-detail.tsx` | Submission summary, cache-hit note, shadcn `Progress` |
| `frontend/src/features/runs/components/predict-wizard.tsx` | Protocol context card, preview panel, commitment button |
| `frontend/src/features/runs/components/prediction-preview.tsx` (new) | The preview panel: count, parse check, thumbnails |
| `frontend/src/features/runs/lib/parse-preview.ts` (new) | Pure: given parsed CSV rows + a column, produce the preview summary |
| `frontend/src/features/runs/lib/parse-preview.test.ts` (new) | Its tests |
| `frontend/src/features/protocols/components/scorecard-view.tsx` | The verdict strip; card grid removed; section order changed |
| `frontend/src/shared/components/chemistry/structure-thumbnail.tsx` | Theme-aware rendering |
| `frontend/src/shared/components/ui/progress.tsx` (new, shadcn CLI) | The progress primitive |

---

## Task 1: Results API — sort, range filters, and `row_id`

**Files:**
- Create: `backend/src/daikonstudio/application/execution/result_view.py`
- Create: `backend/tests/unit/execution/test_result_view.py`
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py:432-540`
- Modify: `backend/src/daikonstudio/interface/routes/runs.py:95-130,171-188`
- Test: `backend/tests/api/test_runs.py` (append)

**Interfaces:**
- Consumes: existing `GetPredictionResults`, `PredictionRow`, `PageResult`, `clamp_limit`, `ValidationError`.
- Produces:
  - `SortSpec(column: str, descending: bool = False)` — frozen dataclass, `kw_only`.
  - `RangeFilter(column: str, minimum: float | None = None, maximum: float | None = None)` — frozen dataclass, `kw_only`.
  - `apply_result_view(frame, *, columns, sort, filters) -> Result[pl.DataFrame, DomainError]` — pure.
  - `ROW_ID = "row_id"` — the column name it mints.
  - `GetPredictionResultsQuery(run_id, cursor, limit, sort: SortSpec | None = None, filters: tuple[RangeFilter, ...] = ())`.
  - `PredictionRow.row_id: int` and `PredictionResponse.row_id: int` on the wire.
  - Query params: `sort_by: str | None`, `sort_dir: Literal["asc","desc"] = "asc"`, `filters: str | None` (URL-encoded JSON).

**Background the implementer needs:**

The results Parquet is written once by the worker and never changes, which is why this endpoint's cursor is a plain integer offset rather than the base64 keyset cursor every other list uses (both surface as `next_cursor` — a real trap). Its columns are `structure`, one column per readout name, `uncertainty`, `applicability`. `uncertainty` is null for every XGBoost row; `applicability` is null when the training set could not be read. The frame is already read fully into memory per page request (a deliberate, ponytail-marked simplification), so filtering and sorting cost nothing extra architecturally.

`row_id` is the correctness core: the client currently derives row identity from `startRow + index`, and `POST /collections` takes those integers as `row_ids` into the original file. Once the server reorders rows, a derived index saves the wrong compounds — silently. Minting `row_id` **before** filter and sort keeps it meaning "position in the original file", which is exactly what `POST /collections` already expects, so that endpoint does not change.

**Why the logic is a separate pure function:** `GetPredictionResults` is I/O orchestration (auth, repositories, blob store), and this repo has no fake repositories — `backend/tests/fakes` holds only `auth.py`, and every existing unit test drives a pure function (`build_scorecard`, `derive_readouts`). Building a fake-repository harness to assert null ordering would be more test scaffolding than the feature. A pure `apply_result_view` over a Polars frame tests the decisions that matter with no harness at all, and the API tests cover the wiring.

- [ ] **Step 1: Write the failing tests for the view function**

Create `backend/tests/unit/execution/test_result_view.py`:

```python
"""Filtering, sorting and stable row identity over a prediction's results.

The frame here deliberately carries nulls in both nullable columns:
`uncertainty` is null for every XGBoost row, and `applicability` is null when
a Protocol's training structures could not be read. Both are real states, and
where they land in a sorted or filtered view is a decision, not an accident.
"""

import polars as pl
import pytest

from daikonstudio.application.execution.result_view import (
    RangeFilter,
    SortSpec,
    apply_result_view,
)

COLUMNS = frozenset({"solubility", "uncertainty", "applicability"})


@pytest.fixture
def frame() -> pl.DataFrame:
    # Row 0 is the least soluble and the least applicable; row 3 carries the
    # nulls. Ordering by any column gives a different permutation, which is
    # what makes the row_id assertions mean something.
    return pl.DataFrame(
        {
            "structure": ["CCO", "CCC", "CCN", "CCF"],
            "solubility": [1.0, 2.0, 3.0, 4.0],
            "uncertainty": [0.5, 0.1, None, 0.3],
            "applicability": [0.2, 0.9, 0.6, None],
        }
    )


def rows(frame, sort=None, filters=()):
    """The row_ids the view yields, in order -- the only assertion that
    survives every reordering, which is the point of having row_id at all."""
    result = apply_result_view(frame, columns=COLUMNS, sort=sort, filters=filters)
    return result.unwrap()["row_id"].to_list()


def test_every_row_carries_its_position_in_the_original_file(frame):
    assert rows(frame) == [0, 1, 2, 3]


def test_sorting_reorders_rows_without_renumbering_them(frame):
    assert rows(frame, sort=SortSpec(column="solubility", descending=True)) == [3, 2, 1, 0]


def test_nulls_sort_last_in_both_directions(frame):
    # A missing measurement is not "the smallest value", and burying it at the
    # top of a descending sort would be the same lie in the other direction.
    assert rows(frame, sort=SortSpec(column="applicability"))[-1] == 3
    assert rows(frame, sort=SortSpec(column="applicability", descending=True))[-1] == 3


def test_a_minimum_excludes_smaller_values_and_nulls(frame):
    assert rows(frame, filters=(RangeFilter(column="applicability", minimum=0.5),)) == [1, 2]


def test_a_maximum_excludes_larger_values_and_nulls(frame):
    assert rows(frame, filters=(RangeFilter(column="uncertainty", maximum=0.4),)) == [1, 3]


def test_both_bounds_keep_only_what_is_between_them(frame):
    assert rows(frame, filters=(RangeFilter(column="solubility", minimum=2.0, maximum=3.0),)) == [
        1,
        2,
    ]


def test_filter_and_sort_compose_and_preserve_row_id(frame):
    assert rows(
        frame,
        filters=(RangeFilter(column="solubility", minimum=2.0),),
        sort=SortSpec(column="solubility", descending=True),
    ) == [3, 2, 1]


def test_a_filter_that_matches_nothing_is_an_empty_view_not_an_error(frame):
    assert rows(frame, filters=(RangeFilter(column="applicability", minimum=1.1),)) == []


def test_an_unknown_sort_column_is_rejected(frame):
    result = apply_result_view(frame, columns=COLUMNS, sort=SortSpec(column="nope"), filters=())
    assert result.failure() is not None


def test_an_unknown_filter_column_is_rejected(frame):
    result = apply_result_view(
        frame, columns=COLUMNS, sort=None, filters=(RangeFilter(column="nope", minimum=1.0),)
    )
    assert result.failure() is not None


def test_the_structure_column_is_not_sortable(frame):
    # Sorting by SMILES string is alphabetical nonsense dressed as chemistry.
    result = apply_result_view(
        frame, columns=COLUMNS, sort=SortSpec(column="structure"), filters=()
    )
    assert result.failure() is not None
```

`Result.failure()` returns `None` on a `Success` in the `returns` library this repo uses, which is what the rejection tests assert against. If that reads awkwardly once written, `assert not is_successful(result)` from `returns.pipeline` is the same assertion — pick one and use it consistently.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cd backend && uv run pytest tests/unit/execution/test_result_view.py -v`
Expected: FAIL — `ModuleNotFoundError: daikonstudio.application.execution.result_view`.

- [ ] **Step 3: Write the view function**

Create `backend/src/daikonstudio/application/execution/result_view.py`:

```python
"""Sorting and range-filtering a prediction Run's results.

Pure, and separate from `GetPredictionResults`, for two reasons. The row
identity decision here is the one that can silently corrupt a Collection, so
it is worth testing without a repository harness in the way. And a results
file is immutable, so a "view" over it is a value, not a query against
changing state -- it belongs to no aggregate and touches no I/O.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.domain.shared.errors import DomainError, ValidationError

__all__ = ["ROW_ID", "RangeFilter", "SortSpec", "apply_result_view"]

#: The row's position in the original results file, which is what
#: `POST /collections` takes as `row_ids`.
ROW_ID = "row_id"


@dataclass(frozen=True, kw_only=True)
class SortSpec:
    column: str
    descending: bool = False


@dataclass(frozen=True, kw_only=True)
class RangeFilter:
    """One column's bounds. Both ends are optional; an entry with neither is
    rejected at the edge, because a filter that filters nothing is a control
    that silently does nothing."""

    column: str
    minimum: float | None = None
    maximum: float | None = None


def apply_result_view(
    frame: pl.DataFrame,
    *,
    columns: Collection[str],
    sort: SortSpec | None,
    filters: Collection[RangeFilter],
) -> Result[pl.DataFrame, DomainError]:
    """`frame` filtered and sorted, with every row stamped with `ROW_ID`.

    `columns` is what may be sorted or filtered on -- the Protocol's readout
    names plus `uncertainty` and `applicability`. `structure` is deliberately
    not among them: ordering compounds by their SMILES string is alphabetical
    nonsense dressed up as chemistry.
    """
    # Minted first, so it always means "position in the original file". Derive
    # it from the page offset instead and a Collection saved under a sort holds
    # different compounds than the ones that were selected, with no symptom.
    view = frame.with_row_index(name=ROW_ID)

    for range_filter in filters:
        if range_filter.column not in columns:
            return Failure(_unknown(range_filter.column, columns, verb="filter on"))
        column = pl.col(range_filter.column)
        # A comparison against null is null, which `filter` drops -- so a
        # bounded column excludes its own nulls without a separate guard. That
        # is intended: "applicability at least 0.5" cannot honestly include a
        # compound whose applicability was never measurable.
        if range_filter.minimum is not None:
            view = view.filter(column >= range_filter.minimum)
        if range_filter.maximum is not None:
            view = view.filter(column <= range_filter.maximum)

    if sort is not None:
        if sort.column not in columns:
            return Failure(_unknown(sort.column, columns, verb="sort by"))
        # Nulls last either way: `uncertainty` is null for every XGBoost row
        # and `applicability` is null when the training set could not be read.
        # Neither is "the smallest value", so neither may lead a page.
        view = view.sort(sort.column, descending=sort.descending, nulls_last=True)

    return Success(view)


def _unknown(column: str, columns: Collection[str], *, verb: str) -> DomainError:
    return ValidationError(
        f"Cannot {verb} '{column}'",
        detail=f"Available columns: {', '.join(sorted(columns))}.",
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd backend && uv run pytest tests/unit/execution/test_result_view.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Use it from the use case**

In `predict_with_protocol.py`:

- Add `row_id: int` to `PredictionRow`, beside `structure`.
- Extend the query object:

```python
@dataclass(frozen=True, kw_only=True)
class GetPredictionResultsQuery:
    run_id: uuid.UUID
    cursor: str | None = None
    limit: int | None = None
    sort: SortSpec | None = None
    filters: tuple[RangeFilter, ...] = ()
```

- Import `ROW_ID`, `RangeFilter`, `SortSpec`, `apply_result_view` from `daikonstudio.application.execution.result_view`.
- In `GetPredictionResults.__call__`, between `frame = pl.read_parquet(io.BytesIO(raw))` and the slice:

```python
        viewed = apply_result_view(
            frame,
            columns={readout.name for readout in protocol.readouts}
            | {"uncertainty", "applicability"},
            sort=query.sort,
            filters=query.filters,
        )
        if not is_successful(viewed):
            return Failure(viewed.failure())
        frame = viewed.unwrap()
```

Import `is_successful` from `returns.pipeline` if the module does not already have an equivalent check — match whatever pattern the file's neighbours use for unwrapping a `Result` (grep the application layer for `is_successful` or `.failure()` before choosing).

- Add `row_id=row[ROW_ID],` to the `PredictionRow(...)` construction.

- [ ] **Step 6: Run the whole backend suite**

Run: `cd backend && uv run pytest`
Expected: PASS. The existing results tests still pass — `PredictionRow` gained a field but no behaviour changed for a query with no sort and no filters.

- [ ] **Step 7: Write the failing HTTP tests**

Append to `backend/tests/api/test_runs.py`, following the file's existing style (`client`, `published_protocol_id`, `prediction_upload_ref`, `_predict`). Add `from urllib.parse import quote` to the imports:

```python
async def test_results_carry_their_row_id(client, published_protocol_id, prediction_upload_ref):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(f"/api/v1/runs/{submitted.json()['id']}/results")
    assert response.status_code == 200, response.text
    assert [item["row_id"] for item in response.json()["items"]] == [0, 1, 2]


async def test_results_can_be_sorted_by_a_readout(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]
    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    readout = protocol["readouts"][0]["name"]

    ascending = await client.get(f"/api/v1/runs/{run_id}/results?sort_by={readout}&sort_dir=asc")
    descending = await client.get(f"/api/v1/runs/{run_id}/results?sort_by={readout}&sort_dir=desc")
    assert ascending.status_code == 200, ascending.text
    assert descending.status_code == 200, descending.text

    def values(response):
        return [item["readouts"][readout]["value"] for item in response.json()["items"]]

    assert values(ascending) == sorted(values(ascending))
    assert values(descending) == list(reversed(values(ascending)))
    # Reordering must not renumber: the same compound keeps the same handle,
    # because that handle is what POST /collections stores.
    assert {item["row_id"] for item in descending.json()["items"]} == {0, 1, 2}


async def test_results_can_be_filtered_by_a_range(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]
    unfiltered = await client.get(f"/api/v1/runs/{run_id}/results")
    assert len(unfiltered.json()["items"]) == 3

    # No compound can be more than perfectly applicable, so this floor empties
    # the view -- which must be a well-formed empty page, not a 4xx.
    response = await client.get(
        f"/api/v1/runs/{run_id}/results?filters=" + quote('{"applicability": {"min": 1.1}}')
    )
    assert response.status_code == 200, response.text
    assert response.json()["items"] == []
    assert response.json()["next_cursor"] is None


async def test_an_unknown_filter_column_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results?filters="
        + quote('{"not_a_column": {"min": 1}}')
    )
    assert response.status_code == 422, response.text


async def test_malformed_filter_json_is_a_422(client, published_protocol_id, prediction_upload_ref):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(f"/api/v1/runs/{submitted.json()['id']}/results?filters=not-json")
    assert response.status_code == 422, response.text


async def test_a_filter_with_no_bounds_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results?filters=" + quote('{"applicability": {}}')
    )
    assert response.status_code == 422, response.text


async def test_an_unknown_sort_direction_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results"
        f"?sort_by={protocol['readouts'][0]['name']}&sort_dir=sideways"
    )
    assert response.status_code == 422, response.text
```

Before running these, confirm the 422 expectation matches how this app renders a `ValidationError`: `interface/error_handlers.py` registers a handler for `DomainError`, and the unknown-column failures travel that path while the malformed-JSON ones travel FastAPI's own. Read that handler and, if `ValidationError` maps to a different status, use that status in the two unknown-column tests rather than forcing the handler to change.

- [ ] **Step 8: Run them and verify they fail**

Run: `cd backend && uv run pytest tests/api/test_runs.py -v -k "row_id or sorted or filtered or filter_column or malformed or no_bounds or sort_direction"`
Expected: FAIL — `row_id` missing, and filter/sort parameters ignored (200 where 422 is expected).

- [ ] **Step 9: Wire the route**

In `runs.py`, add `row_id: int` as the first field of `PredictionResponse` and `row_id=row.row_id` in `from_domain`. Then:

```python
def _parse_filters(raw: str | None) -> tuple[RangeFilter, ...]:
    """`{"applicability": {"min": 0.5}}` -> RangeFilters.

    A JSON object in one query parameter rather than repeated scalar params
    (`applicability_min=...`): filterable columns are a Protocol's own readout
    names, so a fixed parameter list cannot name them without being invented
    per Protocol.
    """
    if raw is None:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise _invalid_filters(f"not valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise _invalid_filters("must be a JSON object")

    filters: list[RangeFilter] = []
    for column, bounds in parsed.items():
        if not isinstance(bounds, dict):
            raise _invalid_filters(f"'{column}' must be an object with min and/or max")
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if minimum is None and maximum is None:
            raise _invalid_filters(f"'{column}' needs at least one of min or max")
        try:
            filters.append(
                RangeFilter(
                    column=column,
                    minimum=None if minimum is None else float(minimum),
                    maximum=None if maximum is None else float(maximum),
                )
            )
        except (TypeError, ValueError) as error:
            raise _invalid_filters(f"'{column}': min and max must be numbers") from error
    return tuple(filters)


def _invalid_filters(message: str) -> RequestValidationError:
    return RequestValidationError(
        [{"loc": ("query", "filters"), "msg": message, "type": "value_error"}]
    )


@router.get("/{run_id}/results", response_model=PaginatedResponse[PredictionResponse])
async def get_run_results(
    run_id: uuid.UUID,
    auth: AuthDep,
    service: GetPredictionResultsDep,
    cursor: str | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    sort_dir: Literal["asc", "desc"] = "asc",
    filters: str | None = None,
) -> PaginatedResponse[PredictionResponse]:
    """`sort_by` and `filters` name columns a Protocol declares, so which names
    are legal is decided in the use case -- this function only parses the wire
    format. `sort_dir` is a Literal, so FastAPI rejects anything else itself."""
    page = result_to_response(
        await service(
            GetPredictionResultsQuery(
                run_id=run_id,
                cursor=cursor,
                limit=limit,
                sort=None
                if sort_by is None
                else SortSpec(column=sort_by, descending=sort_dir == "desc"),
                filters=_parse_filters(filters),
            ),
            auth=auth,
        )
    )
    return PaginatedResponse(
        items=[PredictionResponse.from_domain(row) for row in page.items],
        next_cursor=page.next_cursor,
    )
```

Add imports: `json`, `Literal` from `typing`, `RequestValidationError` from `fastapi.exceptions`, and `RangeFilter`/`SortSpec` from `daikonstudio.application.execution.result_view`. `RequestValidationError` is what FastAPI already renders as 422, so a hand-parsed parameter fails exactly like a declared one.

- [ ] **Step 10: Run the full backend suite**

Run: `cd backend && uv run pytest`
Expected: PASS — everything, including the 285 that already existed.

- [ ] **Step 11: Regenerate the API contract**

Run: `make generate-api`
Expected: `frontend/openapi.json` gains three query parameters and `row_id`; orval rewrites `frontend/src/shared/lib/api/model/predictionResponse.ts`.

Confirm the endpoint count is unchanged (adding parameters must not add paths):

```bash
python3 -c "import json;d=json.load(open('frontend/openapi.json'));print(sum(len(v) for v in d['paths'].values()),'endpoints')"
```

Expected: 21.

- [ ] **Step 12: Commit**

```bash
git add backend/src backend/tests frontend/openapi.json frontend/src/shared/lib/api
git commit -m "feat(api): sort, range filters and stable row ids for run results"
```

---

## Task 2: Triage grid — column filters and the in-domain switch

**Files:**
- Create: `frontend/src/features/runs/lib/result-query.ts`
- Create: `frontend/src/features/runs/lib/result-query.test.ts`
- Modify: `frontend/src/features/runs/hooks/use-runs.ts:99-113`
- Modify: `frontend/src/features/runs/components/triage-grid.tsx`
- Modify: `frontend/src/features/runs/types/index.ts:14-16`

**Interfaces:**
- Consumes: Task 1's `sort_by`/`sort_dir`/`filters` parameters and `PredictionResponse.row_id`.
- Produces:
  - `buildResultParams({sortModel, filterModel, inDomainOnly}): {sort_by?, sort_dir?, filters?}`
  - `fetchResultBlock(runId, startRow, limit, params)` — the fourth argument is `buildResultParams`'s return value.

**Background:** AG Grid's infinite row model hands `getRows` a `sortModel: SortModelItem[]` and a `filterModel` object, and purges its block cache whenever either changes — so a filter change restarts from offset 0, which is what makes the integer cursor safe. AG Grid's number filter emits `{filterType:"number", type:"greaterThanOrEqual"|"lessThanOrEqual"|"inRange", filter: n, filterTo?: n}` per column. The grid is Community edition, so `agNumberColumnFilter` is available but Set filters are not.

- [ ] **Step 1: Write the failing translation tests**

Create `frontend/src/features/runs/lib/result-query.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { buildResultParams } from "./result-query";

describe("AG Grid model -> results API parameters", () => {
  it("sends nothing when nothing is sorted or filtered", () => {
    expect(buildResultParams({ sortModel: [], filterModel: {}, inDomainOnly: false })).toEqual({});
  });

  it("translates a sort", () => {
    expect(
      buildResultParams({
        sortModel: [{ colId: "solubility", sort: "desc" }],
        filterModel: {},
        inDomainOnly: false,
      }),
    ).toEqual({ sort_by: "solubility", sort_dir: "desc" });
  });

  it("takes only the first sort: the API sorts by one column", () => {
    expect(
      buildResultParams({
        sortModel: [
          { colId: "solubility", sort: "asc" },
          { colId: "applicability", sort: "desc" },
        ],
        filterModel: {},
        inDomainOnly: false,
      }),
    ).toEqual({ sort_by: "solubility", sort_dir: "asc" });
  });

  it("translates each number filter operation the columns offer", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          solubility: { filterType: "number", type: "greaterThanOrEqual", filter: 2 },
          uncertainty: { filterType: "number", type: "lessThanOrEqual", filter: 0.5 },
          applicability: { filterType: "number", type: "inRange", filter: 0.2, filterTo: 0.8 },
        },
        inDomainOnly: false,
      }),
    ).toEqual({
      filters: JSON.stringify({
        solubility: { min: 2 },
        uncertainty: { max: 0.5 },
        applicability: { min: 0.2, max: 0.8 },
      }),
    });
  });

  it("turns the in-domain switch into an applicability floor", () => {
    expect(
      buildResultParams({ sortModel: [], filterModel: {}, inDomainOnly: true }),
    ).toEqual({ filters: JSON.stringify({ applicability: { min: 0.5 } }) });
  });

  it("intersects the switch with a user's own applicability filter", () => {
    // Both are active, so both must hold. Taking the tighter bound is the only
    // reading that never shows a compound the switch says to hide.
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          applicability: { filterType: "number", type: "inRange", filter: 0.1, filterTo: 0.7 },
        },
        inDomainOnly: true,
      }),
    ).toEqual({ filters: JSON.stringify({ applicability: { min: 0.5, max: 0.7 } }) });
  });

  it("keeps the user's floor when it is already tighter than the switch", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: {
          applicability: { filterType: "number", type: "greaterThanOrEqual", filter: 0.9 },
        },
        inDomainOnly: true,
      }),
    ).toEqual({ filters: JSON.stringify({ applicability: { min: 0.9 } }) });
  });

  it("drops a filter with no usable bound rather than sending an empty one", () => {
    expect(
      buildResultParams({
        sortModel: [],
        filterModel: { solubility: { filterType: "number", type: "greaterThanOrEqual" } },
        inDomainOnly: false,
      }),
    ).toEqual({});
  });
});
```

- [ ] **Step 2: Run and verify failure**

Run: `cd frontend && pnpm test result-query`
Expected: FAIL — cannot resolve `./result-query`.

- [ ] **Step 3: Write the translation**

Create `frontend/src/features/runs/lib/result-query.ts`:

```typescript
import type { SortModelItem } from "ag-grid-community";

/** The applicability below which a compound is extrapolation, not prediction. */
export const IN_DOMAIN_FLOOR = 0.5;

export interface ResultParams {
  sort_by?: string;
  sort_dir?: "asc" | "desc";
  filters?: string;
}

interface NumberFilter {
  filterType?: string;
  type?: string;
  filter?: number | null;
  filterTo?: number | null;
}

interface Bounds {
  min?: number;
  max?: number;
}

function boundsFor(model: NumberFilter): Bounds | null {
  const { type, filter, filterTo } = model;
  if (type === "greaterThanOrEqual" && filter != null) return { min: filter };
  if (type === "lessThanOrEqual" && filter != null) return { max: filter };
  if (type === "inRange" && filter != null && filterTo != null)
    return { min: filter, max: filterTo };
  // Any other operation is one the columns do not offer and the API cannot
  // honour; sending it would filter server-side by something else entirely.
  return null;
}

/**
 * AG Grid's sort and filter models, plus the in-domain switch, as query
 * parameters for `GET /runs/{id}/results`.
 *
 * Pure and separately tested because it is the one place where a client-side
 * control becomes a server-side promise: a mistranslation here shows a
 * chemist a filtered grid that was filtered by something else.
 */
export function buildResultParams({
  sortModel,
  filterModel,
  inDomainOnly,
}: {
  sortModel: SortModelItem[];
  filterModel: Record<string, NumberFilter>;
  inDomainOnly: boolean;
}): ResultParams {
  const params: ResultParams = {};

  // Single-column sort: the API sorts by one column, and pretending otherwise
  // would silently drop the rest of the user's intent.
  const [primary] = sortModel;
  if (primary) {
    params.sort_by = primary.colId;
    params.sort_dir = primary.sort === "desc" ? "desc" : "asc";
  }

  const filters: Record<string, Bounds> = {};
  for (const [column, model] of Object.entries(filterModel ?? {})) {
    const bounds = boundsFor(model);
    if (bounds) filters[column] = bounds;
  }

  if (inDomainOnly) {
    // Both constraints hold at once, so the bounds intersect: the tighter
    // floor wins, and the user's own ceiling survives.
    const existing = filters.applicability ?? {};
    filters.applicability = {
      ...existing,
      min: Math.max(existing.min ?? IN_DOMAIN_FLOOR, IN_DOMAIN_FLOOR),
    };
  }

  if (Object.keys(filters).length > 0) params.filters = JSON.stringify(filters);
  return params;
}
```

- [ ] **Step 4: Run and verify the tests pass**

Run: `cd frontend && pnpm test result-query`
Expected: PASS (8 tests).

- [ ] **Step 5: Take `__rowId` from the API**

In `frontend/src/features/runs/hooks/use-runs.ts`, replace `fetchResultBlock` (its docstring's note about the integer cursor stays true and stays):

```typescript
export async function fetchResultBlock(
  runId: string,
  startRow: number,
  limit: number,
  params: ResultParams,
): Promise<{ rows: TriageRow[]; nextCursor: string | null }> {
  const page = await customInstance<PaginatedResponsePredictionResponse>({
    url: `${API_V1}/runs/${runId}/results`,
    method: "GET",
    params: { cursor: String(startRow), limit, ...params },
  });
  return {
    // `row_id` from the server, never the page offset: under a sort or filter
    // the two disagree, and the offset would save the wrong compounds into a
    // Collection without any visible symptom.
    rows: page.items.map((item) => ({ ...item, __rowId: item.row_id })),
    nextCursor: page.next_cursor ?? null,
  };
}
```

Import `ResultParams` from `../lib/result-query`. In `frontend/src/features/runs/types/index.ts`, `TriageRow` keeps `__rowId` (AG Grid's `getRowId` contract) — leave it as is.

- [ ] **Step 6: Wire the grid**

In `triage-grid.tsx`:

- Add `const [inDomainOnly, setInDomainOnly] = useState(false);`
- Define the shared numeric-column filter config at module scope, beside `BLOCK_SIZE`:

```typescript
const NUMBER_FILTER = {
  sortable: true,
  filter: "agNumberColumnFilter" as const,
  filterParams: {
    // Only what the API can honour. Offering "not equal" or "blank" would be
    // a control that quietly filters by something else.
    filterOptions: ["greaterThanOrEqual", "lessThanOrEqual", "inRange"],
    maxNumConditions: 1,
    buttons: ["reset"],
  },
} satisfies Partial<ColDef<TriageRow>>;
```

- Spread it into each numeric column definition — every readout column, plus uncertainty and applicability — e.g. the readout loop becomes:

```typescript
      base.push({
        headerName: readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
        // A valueGetter column has no `field` to derive a colId from, and the
        // colId is what the API receives as the column name to sort by.
        colId: readout.name,
        width: 150,
        ...NUMBER_FILTER,
        valueGetter: (params) => params.data?.readouts?.[readout.name]?.value ?? null,
        cellRenderer: (params: { value: number | null }) => (
          <ReadoutValue value={params.value} unit={readout.unit} precision={3} />
        ),
      });
```

  Uncertainty and applicability already have `field`, which supplies their `colId`; they need only the spread.

- Give the structure and SMILES columns `sortable: false, filter: false` (the structure column already has both). Ordering compounds by their SMILES string is alphabetical nonsense dressed as chemistry, and the API rejects it anyway.
- The datasource passes the translated params through and depends on the switch:

```typescript
  const datasource = useMemo<IDatasource>(
    () => ({
      rowCount: undefined,
      getRows: async (params) => {
        try {
          const { rows, nextCursor } = await fetchResultBlock(
            runId,
            params.startRow,
            params.endRow - params.startRow,
            buildResultParams({
              sortModel: params.sortModel,
              filterModel: params.filterModel ?? {},
              inDomainOnly,
            }),
          );
          const lastRow = nextCursor === null ? params.startRow + rows.length : undefined;
          params.successCallback(rows, lastRow);
        } catch {
          params.failCallback();
        }
      },
    }),
    [runId, inDomainOnly],
  );
```

- Because `datasource` is only handed to the grid in `onGridReady`, flipping the switch needs an explicit re-set. Add:

```typescript
  useEffect(() => {
    // AG Grid purges its block cache when the sort or filter model changes,
    // but the in-domain switch lives outside both -- re-setting the datasource
    // is what makes it restart from offset 0 instead of appending a filtered
    // page onto unfiltered blocks.
    apiRef.current?.setGridOption("datasource", datasource);
  }, [datasource]);
```

- Replace the "no filter box" comment block with the switch, in the same header row and left of the selection summary:

```tsx
        <div className="flex items-center gap-2">
          <Switch id="in-domain" checked={inDomainOnly} onCheckedChange={setInDomainOnly} />
          <Label htmlFor="in-domain" className="text-sm font-normal">
            In domain only
          </Label>
        </div>
```

Import `Switch` and `Label` from `@/shared/components/ui/*`, `useEffect` from React, and `buildResultParams` from `../lib/result-query`.

- [ ] **Step 7: Verify types, lint and the whole frontend suite**

Run: `cd frontend && pnpm exec tsc --noEmit && pnpm lint && pnpm test`
Expected: all three exit 0. Check exit codes; do not eyeball the output.

- [ ] **Step 8: Verify in a browser**

Restart the backend first (`make dev-be`) — a stale uvicorn still serves the old schema and the grid will look broken for the wrong reason. Then open an existing prediction run at `http://localhost:3003/runs/<id>`:

1. Sort by a readout — both directions, and confirm the rows reorder.
2. Filter applicability with in-range, confirm the row count shrinks.
3. Flip "In domain only" and confirm the grid restarts from the top.
4. Select rows under a filter, save as a collection, and open the collection: the compounds in it must be the ones that were selected. This is the `row_id` check and it is the one that matters.
5. Read the browser console: AG Grid warns loudly about misconfiguration, and those warnings are how the infinite-model traps announce themselves.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/features/runs
git commit -m "feat(runs): server-side sorting and filtering in the triage grid"
```

---

## Task 3: Scorecard — the verdict strip

**Files:**
- Create: `frontend/src/shared/components/ui/progress.tsx` (via shadcn CLI)
- Modify: `frontend/src/features/protocols/components/scorecard-view.tsx`

**Interfaces:**
- Consumes: `computeVerdict`, `computeOptimismGap`, `metricLabel` — all unchanged.
- Produces: nothing other tasks depend on.

**Background:** This is a layout change; `lib/verdict.ts` and its tests are not touched. The rule the layout enforces: the verdict cannot be read without its caveats. ESOL is the case that proves it — RMSE 1.200 against a baseline of 1.228, with an optimism gap of 0.165 and a noise floor of 0.151, meaning the model's entire edge is a fifth of the measurement error. The page is supposed to look uncomfortable when the model is not good.

- [ ] **Step 1: Add the shadcn Progress primitive**

Run: `cd frontend && pnpm dlx shadcn@latest add progress`
Expected: creates `src/shared/components/ui/progress.tsx` (the `components.json` aliases put it there) and installs `@radix-ui/react-progress`.

Verify it landed in the right place and nothing else changed:

```bash
git status --short frontend/src/shared/components/ui frontend/package.json
```

- [ ] **Step 2: Fuse the honesty numbers into the verdict band**

In `scorecard-view.tsx`, replace `OptimismGapCard` and the `HonestyCard` usages with a stat row rendered inside `VerdictBand`. Add above `VerdictBand`:

```tsx
function HonestyStat({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-[9rem] flex-1">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

/**
 * The three numbers that argue with the verdict, inside the verdict's own
 * border.
 *
 * They were three separate cards below the band, and separate blocks can be
 * read separately: the ESOL run showed a green "beats the baseline" while an
 * optimism gap six times the winning margin sat in a card underneath it. In
 * one border, nobody reads the claim without the doubts. The captions stay
 * visible for the same reason -- hover-to-see-the-caveat is a way of hiding
 * one.
 */
function HonestyStats({ scorecard }: { scorecard: ScorecardResponse }) {
  const gap = computeOptimismGap(scorecard);
  const coverage = scorecard.applicability_coverage;
  const metric = metricLabel(scorecard.primary_metric);

  return (
    <div className="mt-4 flex flex-wrap gap-x-8 gap-y-4 border-t border-current/15 pt-3">
      <HonestyStat label="Optimism gap">
        {gap.kind === "shown" ? (
          <>
            <ReadoutValue value={gap.gap} precision={3} className="text-xl font-semibold" />
            <p className="mt-1 text-xs text-muted-foreground">
              {metric} was <ReadoutValue value={gap.random} precision={3} /> on a random split and{" "}
              <ReadoutValue value={gap.scaffold} precision={3} /> on the scaffold split it was
              actually scored on — the difference an easier split would have flattered it by.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">{gap.message}</p>
        )}
      </HonestyStat>

      {/* Absent, not empty, for a binary target: there are no replicate
          spreads to average, so the question does not arise. */}
      {scorecard.noise_floor != null && (
        <HonestyStat label="Assay noise floor">
          <ReadoutValue
            value={scorecard.noise_floor}
            unit={scorecard.unit}
            precision={3}
            className="text-xl font-semibold"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Repeat measurements of the same compound disagreed by this much. No model trained on
            this data can honestly do better.
          </p>
        </HonestyStat>
      )}

      <HonestyStat label="Applicability">
        {coverage == null ? (
          <p className="text-xs text-muted-foreground">Could not be computed for this protocol.</p>
        ) : (
          <>
            <span className="text-xl font-semibold tabular-nums">
              {(coverage * 100).toFixed(0)}%
            </span>
            <Progress value={coverage * 100} className="mt-1.5 h-1.5" />
            <p className="mt-1 text-xs text-muted-foreground">
              of test compounds sit close enough to the training set for the model to have seen
              anything like them. The rest is extrapolation.
            </p>
          </>
        )}
      </HonestyStat>
    </div>
  );
}
```

Render `<HonestyStats scorecard={scorecard} />` inside `VerdictBand`'s outer `div`, as the last child of the non-`is-baseline`, non-`unknown` branch's fragment — after the existing "The baseline is …" paragraph. The `is-baseline` and `unknown` branches keep exactly their current copy and get no stat row: there is no comparison to cross-examine.

- [ ] **Step 3: Reorder the page and delete the card grid**

Replace `ScorecardView`'s body:

```tsx
export function ScorecardView({ scorecard }: { scorecard: ScorecardResponse }) {
  return (
    <div className="space-y-4">
      <VerdictBand scorecard={scorecard} />
      {/* Before the metric table: the ESOL run's most actionable finding was
          that 8 of its 20 worst predictions had no ring system at all. An
          aggregate cannot say that, and a table of aggregates should not
          outrank it. */}
      <WorstRows scorecard={scorecard} />
      <MetricTable scorecard={scorecard} />
    </div>
  );
}
```

Delete `OptimismGapCard` and `HonestyCard` entirely, along with the now-unused `Card`/`CardContent`/`CardHeader`/`CardTitle` imports if `MetricTable` and `WorstRows` no longer need them (they do — keep the imports). Import `Progress` from `@/shared/components/ui/progress`.

- [ ] **Step 4: Use Progress for the run's own bar**

In `frontend/src/features/runs/components/run-detail.tsx`, replace the hand-rolled bar:

```tsx
            <Progress value={Math.round(run.progress * 100)} />
```

- [ ] **Step 5: Verify types, lint and tests**

Run: `cd frontend && pnpm exec tsc --noEmit && pnpm lint && pnpm test`
Expected: all exit 0. `verdict.test.ts` still passes untouched — if it does not, the layout change reached into logic it should not have.

- [ ] **Step 6: Verify in a browser**

Open the ESOL protocol's Scorecard at `http://localhost:3003/protocols/<id>`:

- The verdict headline, the metric comparison and all three honesty numbers are inside one border.
- The order is verdict → Where it fails → All metrics.
- Resize narrow: the stats wrap rather than overflow, and the band stays readable.
- Toggle light/dark: the hairline separator and the progress bar are visible in both.

- [ ] **Step 7: Commit**

```bash
git add frontend/src
git commit -m "feat(protocols): fuse the scorecard's honesty numbers into the verdict"
```

---

## Task 4: Predict flow — preview, protocol context, honest progress

**Files:**
- Create: `frontend/src/features/runs/lib/parse-preview.ts`
- Create: `frontend/src/features/runs/lib/parse-preview.test.ts`
- Create: `frontend/src/features/runs/components/prediction-preview.tsx`
- Modify: `frontend/src/features/runs/components/predict-wizard.tsx`
- Modify: `frontend/src/features/runs/components/run-detail.tsx`
- Modify: `frontend/src/features/runs/index.ts` (barrel, if the new component is exported — it is not; it is internal to the feature)

**Interfaces:**
- Consumes: `useProtocol`, `useDataset`, `useEngines`, `ConditionSummary`, `StructureThumbnail`, `Progress`.
- Produces: `summarisePreview(rows, column)` → `{ total: number; sample: string[]; blank: number }`.

**Background, all verified in code — do not re-derive:**

1. **Unparseable structures are dropped, not fatal.** `RunPrediction.__call__` canonicalises every structure and keeps only the ones that parse (`valid_frame = frame.filter(is_valid)`); it raises only when *nothing* parses ("No valid structures in the uploaded file"). So the preview's copy must say those rows are skipped and the rest are still scored — not that the run will fail.
2. **`DatasetResponse.row_count` is the honest compound count.** It is `split_frame.height` — after replicates are grouped (997 for ESOL, where `valid_rows` says 1,008). `valid_rows` remains disqualified for any compound claim.
3. **Conditions are read-only here.** Both engines ignore `ctx.conditions` at predict time — `predict()` delegates to `_predict_with_tree_ensemble`, which never reads them. Editable fields would be dead controls. `ConditionSummary` already renders an engine's conditions as a description list and is exported from `@/features/engines`.
4. **A cache hit returns 202 with an already-`ready` Run.** `PredictWithProtocol` returns the existing Run when one matches the cache key and is READY. Branching on the returned status is the only way to tell.

- [ ] **Step 1: Write the failing preview-summary tests**

Create `frontend/src/features/runs/lib/parse-preview.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { PREVIEW_SAMPLE_SIZE, summarisePreview } from "./parse-preview";

const rows = [
  { smiles: "CCO", note: "a" },
  { smiles: "CCC", note: "b" },
  { smiles: "", note: "c" },
  { smiles: "  ", note: "d" },
];

describe("prediction upload preview", () => {
  it("counts every row in the file", () => {
    expect(summarisePreview(rows, "smiles").total).toBe(4);
  });

  it("counts blank cells separately: they are not structures", () => {
    expect(summarisePreview(rows, "smiles").blank).toBe(2);
  });

  it("samples the first few non-blank structures for rendering", () => {
    expect(summarisePreview(rows, "smiles").sample).toEqual(["CCO", "CCC"]);
  });

  it("caps the sample", () => {
    const many = Array.from({ length: 50 }, (_, index) => ({ smiles: `C${"C".repeat(index)}` }));
    expect(summarisePreview(many, "smiles").sample).toHaveLength(PREVIEW_SAMPLE_SIZE);
  });

  it("reports an empty summary for a column that is not there", () => {
    expect(summarisePreview(rows, "nope")).toEqual({ total: 4, sample: [], blank: 4 });
  });
});
```

- [ ] **Step 2: Run and verify failure**

Run: `cd frontend && pnpm test parse-preview`
Expected: FAIL — cannot resolve `./parse-preview`.

- [ ] **Step 3: Write it**

Create `frontend/src/features/runs/lib/parse-preview.ts`:

```typescript
/** How many structures the preview draws. Enough to recognise the set. */
export const PREVIEW_SAMPLE_SIZE = 6;

export interface PreviewSummary {
  /** Every data row in the file, blank structure cells included. */
  total: number;
  /** The first few non-blank structures, for thumbnails. */
  sample: string[];
  /** Rows whose structure cell is empty. */
  blank: number;
}

/**
 * What a dropped CSV contains, before anything is uploaded.
 *
 * Counting happens here rather than in the component so the number on the Run
 * button -- the one the user is asked to commit to -- is covered by a test.
 */
export function summarisePreview(
  rows: Record<string, string | undefined>[],
  column: string,
): PreviewSummary {
  const sample: string[] = [];
  let blank = 0;

  for (const row of rows) {
    const value = row[column]?.trim() ?? "";
    if (value === "") {
      blank += 1;
      continue;
    }
    if (sample.length < PREVIEW_SAMPLE_SIZE) sample.push(value);
  }

  return { total: rows.length, sample, blank };
}
```

- [ ] **Step 4: Run and verify the tests pass**

Run: `cd frontend && pnpm test parse-preview`
Expected: PASS (5 tests).

- [ ] **Step 5: Build the preview panel**

Create `frontend/src/features/runs/components/prediction-preview.tsx`:

```tsx
"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import type { PreviewSummary } from "../lib/parse-preview";

/**
 * What was actually read out of the dropped file.
 *
 * The point is to answer "is this the right column?" before a run is
 * submitted rather than after: the structures render from the column that is
 * currently selected, so picking the wrong one looks wrong immediately.
 */
export function PredictionPreview({
  summary,
  column,
}: {
  summary: PreviewSummary;
  column: string;
}) {
  const usable = summary.total - summary.blank;

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3">
      <p className="text-sm">
        <span className="font-medium">
          {usable} compound{usable === 1 ? "" : "s"}
        </span>{" "}
        <span className="text-muted-foreground">
          in <span className="font-mono text-xs">{column}</span>
        </span>
      </p>

      {summary.blank > 0 && (
        // A forecast about what will happen, so it sits outside any button:
        // the worker drops rows it cannot read and scores the rest, and this
        // is the only place that is visible before submitting.
        <p className="mt-1 text-xs text-warning">
          {summary.blank} row{summary.blank === 1 ? " has" : "s have"} no structure and will be
          skipped.
        </p>
      )}

      {summary.sample.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {summary.sample.map((smiles) => (
            <StructureThumbnail key={smiles} smiles={smiles} size={56} />
          ))}
        </div>
      )}
    </div>
  );
}
```

`StructureThumbnail` already renders its own "no structure" state for a SMILES RDKit cannot parse, so an unparseable structure in the sample shows as one — no separate parse pass is needed, and no second RDKit call per row.

- [ ] **Step 6: Wire the wizard**

In `predict-wizard.tsx`:

- Parse the whole file, not five rows. In `onDrop`, drop `preview: 5` and keep the parsed rows:

```typescript
      complete: (result) => {
        const fields = (result.meta.fields ?? []).filter((field) => field.trim() !== "");
        if (fields.length === 0) {
          showError("No columns found. Is this a CSV with a header row?");
          return;
        }
        setFile(dropped);
        setColumns(fields);
        setRows(result.data);
        const guess =
          fields.find((field) => field.toLowerCase().trim() === "smiles") ??
          fields.find((field) => field.toLowerCase().includes("smiles")) ??
          fields[0];
        setStructureColumn(guess);
      },
```

with `const [rows, setRows] = useState<Record<string, string | undefined>[]>([]);`

- Derive the summary and the count:

```typescript
  const summary = useMemo(
    () => (rows.length > 0 && structureColumn ? summarisePreview(rows, structureColumn) : null),
    [rows, structureColumn],
  );
  const compoundCount = summary ? summary.total - summary.blank : 0;
```

- Add the protocol context card. The selected protocol comes from the already-loaded list; the dataset supplies the honest training count and the engine supplies condition labels:

```tsx
function ProtocolContext({ protocol }: { protocol: Protocol }) {
  const { data: dataset } = useDataset(protocol.dataset_id);
  const { data: engines } = useEngines();
  const engine = engines?.find((candidate) => candidate.id === protocol.engine_id);
  const conditions = engine?.conditions ?? [];

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3 text-sm">
      <p>
        {/* A classification Protocol declares two readouts (probability and
            class), so this joins rather than concatenating them into one
            unreadable run-on. */}
        Predicts{" "}
        <span className="font-medium">
          {protocol.readouts
            .map((readout) => (readout.unit ? `${readout.name} (${readout.unit})` : readout.name))
            .join(" and ")}
        </span>
        {dataset && (
          <span className="text-muted-foreground">
            {" "}
            · trained on {dataset.row_count} compounds from {dataset.name}
          </span>
        )}
      </p>
      {conditions.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Trained with
          </p>
          {/* Read-only, and not an oversight: neither engine reads conditions
              at predict time, so an input here would be a control that changes
              nothing. When an engine declares predict-time conditions, this
              becomes ConditionFields. */}
          <dl className="mt-1 flex flex-wrap gap-x-6 gap-y-1 text-xs">
            {conditions.map((condition) => (
              <div key={condition.key} className="flex gap-1.5">
                <dt className="text-muted-foreground">{condition.label}</dt>
                <dd className="font-mono">
                  {String(protocol.conditions?.[condition.key] ?? condition.default ?? "—")}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}
```

Render it under the protocol `Select` when one is chosen (`published.find((p) => p.id === protocolId)`), and render `<PredictionPreview summary={summary} column={structureColumn} />` under the structure-column select when `summary` is non-null. Imports: `useDataset` from `@/features/datasets`, `useEngines` from `@/features/engines`, `Protocol` type from `@/features/protocols`, `useMemo` from React.

- `useDataset` and `useEngines` are exported from their feature barrels already; if `useDataset` is not in `@/features/datasets`'s `index.ts`, add it there rather than importing through a deep path.

- The Run button states the commitment:

```tsx
        <Button onClick={submit} disabled={!protocolId || !file || compoundCount === 0 || busy}>
          {busy ? "Starting…" : `Score ${compoundCount} compound${compoundCount === 1 ? "" : "s"}`}
        </Button>
```

- Carry the count and the cache state through the navigation:

```typescript
      const run = await create.mutateAsync({
        protocol_id: protocolId,
        upload_ref: uploadRef,
        structure_column: structureColumn,
      });
      // A cache hit comes back 202 with an already-ready Run, so the status is
      // the only way to tell that no work was started. Saying so beats showing
      // a progress bar that was never going to move.
      const cached = run.status === "ready" ? "&cached=1" : "";
      router.push(`/runs/${run.id}?compounds=${compoundCount}${cached}`);
```

- [ ] **Step 7: Show the submission summary and the cache note**

In `run-detail.tsx`, read the parameters and render them:

```tsx
  const params = useSearchParams();
  const submittedCount = params.get("compounds");
  const fromCache = params.get("cached") === "1";
```

Under the run's timestamp:

```tsx
          {submittedCount && (
            <p className="mt-1 text-sm text-muted-foreground">
              {submittedCount} compound{submittedCount === "1" ? "" : "s"} submitted
            </p>
          )}
```

And above the grid, in place of nothing (the progress card renders only while `running`, and a cached run is already `ready`, so the two never collide):

```tsx
      {fromCache && run.status === "ready" && (
        <div className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
          These compounds had already been scored by this protocol — these results came from cache,
          not a new run.
        </div>
      )}
```

Absence stays absence: a run opened from the list has neither parameter and shows neither line. No backend change is spent on making them always available.

- [ ] **Step 8: Verify types, lint and tests**

Run: `cd frontend && pnpm exec tsc --noEmit && pnpm lint && pnpm test`
Expected: all exit 0.

- [ ] **Step 9: Verify in a browser**

At `http://localhost:3003/runs/new`, with `esol-holdout.csv` (120 structures) from `~/Downloads` or this session's scratchpad:

1. Choose the ESOL protocol: the context card names the readout and unit and shows the trained-with conditions, and the training count is 997 — not 1,008.
2. Drop the CSV: the preview shows 120 compounds and six structures, and the button reads "Score 120 compounds".
3. Switch the structure column to a non-structure column: the thumbnails become "no structure" boxes, which is the point.
4. Submit: run detail shows "120 compounds submitted" and a moving progress bar.
5. Submit the identical file again: the cache note appears instead of a progress bar, and results are there immediately.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/features/runs
git commit -m "feat(runs): preview compounds and protocol context before a prediction"
```

---

## Task 5: Theme-aware structure rendering

**Files:**
- Modify: `frontend/src/shared/components/chemistry/structure-thumbnail.tsx`

**Interfaces:**
- Consumes: `useTheme` from `next-themes` (the provider uses `attribute="data-theme"`, `defaultTheme="dark"`, `enableSystem`).
- Produces: no API change — same props.

**Background:** The current `dark:invert` puts every structure on a black rectangle (the SVG has a white background) and inverts heteroatom colours into their complements, so oxygen reads cyan. The installed RDKit (`@rdkit/rdkit` 2025.3.4-1.0.0) accepts `backgroundColour`, `clearBackground` and `atomColourPalette` in the drawing-options JSON — verified against the shipped wasm, not assumed. Colours are `[r, g, b]` floats in 0–1.

- [ ] **Step 1: Render with a themed palette**

Replace the drawing call and the `<img>` class. Add at module scope:

```typescript
/**
 * A palette for a dark background.
 *
 * Keyed by atomic number, RDKit's own convention: 0 is the default for
 * anything not listed. Carbon and hydrogen go light so bonds read on a dark
 * surface; the heteroatom colours stay near RDKit's defaults, because a
 * chemist reads oxygen as red and nitrogen as blue and re-teaching that would
 * be a worse crime than a slightly dim red.
 */
const DARK_PALETTE = {
  0: [0.9, 0.9, 0.9],
  1: [0.9, 0.9, 0.9],
  6: [0.9, 0.9, 0.9],
  7: [0.4, 0.55, 1.0],
  8: [1.0, 0.4, 0.4],
  9: [0.4, 0.9, 0.5],
  15: [1.0, 0.6, 0.3],
  16: [0.95, 0.85, 0.3],
  17: [0.4, 0.9, 0.5],
  35: [0.85, 0.55, 0.35],
  53: [0.7, 0.5, 0.9],
};
```

In the component, take the resolved theme and include it in the effect's dependencies:

```typescript
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
```

```typescript
          const svg = mol.get_svg_with_highlights(
            JSON.stringify({
              width: size * 2,
              height: size * 2,
              bondLineWidth: 1.6,
              minFontSize: 13,
              addAtomIndices: false,
              // Transparent, so a structure sits on whatever surface holds it
              // -- a card, a grid row, a dialog -- instead of on a white
              // rectangle that has to be inverted back out in dark mode.
              clearBackground: false,
              ...(isDark ? { atomColourPalette: DARK_PALETTE } : {}),
            }),
          );
```

with `}, [smiles, size, isDark]);` and the `<img>` class becoming `className={cn(className)}` — `dark:invert` is gone.

Import `useTheme` from `next-themes`.

- [ ] **Step 2: Verify types, lint and tests**

Run: `cd frontend && pnpm exec tsc --noEmit && pnpm lint && pnpm test`
Expected: all exit 0.

- [ ] **Step 3: Verify in a browser, in both themes**

Open a run's triage grid and a Scorecard's "Where it fails" section, then toggle the theme in settings:

- No black or white rectangles behind any structure in either theme.
- Bonds are legible on both surfaces; heteroatoms are still recognisable by colour.
- Toggling the theme re-renders existing thumbnails rather than leaving stale ones (the effect depends on `isDark`).
- If the dark palette reads badly against the app's surface, tune `DARK_PALETTE` here — it is one object, and this is the calibration knob for it.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/shared/components/chemistry/structure-thumbnail.tsx
git commit -m "fix(chemistry): render structures with a themed palette, not an inversion"
```

---

## Task 6: Live verification of the two unexercised paths

**Files:** none — this task changes no code unless it finds a defect.

**Background:** Both paths have unit tests and neither has been seen in a browser. They matter now because both exercise the verdict strip's absent-states, which Task 3 rebuilt: a binary target has no noise floor to show, and a `baseline_is_self` protocol has no comparison to render. An absent state that renders as an empty box instead of vanishing is exactly the bug this pass exists to prevent.

Test data is in `~/Downloads` and this session's scratchpad; re-fetch from `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/` if gone.

- [ ] **Step 1: Confirm the backend is current**

```bash
curl -s localhost:8002/openapi.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(len(v) for v in d['paths'].values()), 'endpoints')"
```

Expected: 21. A stale uvicorn is the single most expensive trap in this repo — every symptom points at the frontend.

- [ ] **Step 2: Binary classification, end to end**

Upload `bace-active.csv` (1513 rows, binary), freeze the Dataset, train with **ecfp4-xgboost**, and open the Scorecard:

- MCC and balanced accuracy lead the metric table.
- **The noise-floor stat is absent from the verdict strip** — not an empty box, not an em dash where a caption should be. Replicate spreads do not exist for a binary target, so the question does not arise.
- The optimism gap and applicability stats are present and readable.
- Publish it, run it over a slice of the same file with the target column removed, and confirm the triage grid shows a probability and a class column, each with its own unit.

- [ ] **Step 3: `baseline_is_self`**

Train a second protocol on the same dataset with **ecfp4-randomforest** — the baseline engine — and open its Scorecard:

- The verdict says the model *is* the baseline; no comparison is rendered anywhere on the page.
- The metric table's third column reads "is the baseline" rather than showing numbers.
- No stat row of honesty numbers appears under the verdict (there is no comparison to cross-examine).

- [ ] **Step 4: Record what was found**

If both paths are clean, note it in the handoff (Task 7). If either shows a defect, stop and fix it with a test first — a rendering bug in an absent-state is the exact failure this pass is about, and shipping it would be worse than shipping nothing.

---

## Task 7: Update the handoff

**Files:**
- Modify: `docs/superpowers/HANDOFF-frontend-phase-2.md`

- [ ] **Step 1: Bring §3 and §4 up to date**

Rewrite "What is NOT done": remove the UX polish pass, sorting/filtering run results, and dark-mode structure rendering (all now done); keep Playwright, the dashboard, and the Phase 2 product features. Add to the traps section, in the file's existing voice:

- The results cursor is still a plain integer offset behind the same `next_cursor` field name as everything else's base64 keyset cursor — and it now means "offset within the current filtered and sorted view", which is why a filter change must restart at zero.
- `row_id` comes from the server and is the row's position in the original results file; deriving it from the page offset breaks silently under sort or filter, and the symptom is a Collection holding the wrong compounds.
- Both engines ignore `ctx.conditions` at predict time, which is why the predict wizard shows the trained-with conditions read-only rather than as inputs.
- Whether the two paths in Task 6 came out clean, and anything they exposed.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/HANDOFF-frontend-phase-2.md
git commit -m "docs: handoff after the Phase 2 UX pass"
```

---

## Verification Summary

Before calling this done, all of these must hold — by exit code, not by reading output:

| Check | Command | Expected |
|---|---|---|
| Backend suite | `cd backend && uv run pytest` | pass, 285 existing + 18 new (11 unit, 7 API) |
| Frontend types | `cd frontend && pnpm exec tsc --noEmit` | exit 0 |
| Frontend lint | `cd frontend && pnpm lint` | exit 0 |
| Frontend tests | `cd frontend && pnpm test` | pass, 15 existing + 13 new |
| API snapshot | endpoint count from `frontend/openapi.json` | 21 |
| Loop | browser, end to end | upload → freeze → train → scorecard → publish → predict → triage → collection → export |
| Absent states | browser, Task 6 | binary: no noise floor; `baseline_is_self`: no comparison |
