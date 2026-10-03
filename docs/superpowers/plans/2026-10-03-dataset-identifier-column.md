# Dataset Identifier Column Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dataset can name its identifier column (at upload or later, by any editor), and each training compound's ID appears in the Compounds tab (with search), the scorecard's largest errors, and chemical-space map tooltips.

**Architecture:** `datasets.id_column` (migration 012) records which snapshot column holds IDs; nothing else is copied. One helper reads a structure → ID map from the snapshot at read time (snapshot structures are canonical and unique), used by the scorecard and map lookups; the compounds endpoint reads the column directly. The frontend adds the picker to the wizard and the Overview tab, and renders IDs where compounds appear.

**Tech Stack:** FastAPI, SQLAlchemy async, Alembic, polars, pytest; Next 16, React 19, TanStack Query v5, Radix Select, vitest.

**Spec:** `docs/superpowers/specs/2026-10-03-dataset-identifier-column-design.md`

## Global Constraints

- Any editor may set or change the identifier column; viewers may not (403).
- Allowed: any snapshot column except the structure column, the target column and `split`.
- Messages: "Choose an identifier column other than the structure, target or split column." / "Column '{name}' is not in the uploaded file." / "This dataset has no identifier column."
- `id_column` is not part of `content_hash`.
- IDs are looked up at read time, never copied into scorecard inputs or map files. Blank ID values read as no ID.
- Search is a case-insensitive "contains" match on the ID text.
- UI copy American English, academic and plain; no em-dash clause chains.
- Backend gates from `backend/`: `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`, `uv run lint-imports`, `env OMP_NUM_THREADS=1 uv run pytest -q`.
- Frontend gates from `frontend/`: `pnpm lint`, `pnpm exec tsc --noEmit`, `pnpm test`. Run `pnpm exec biome check --write src` before `pnpm lint`.
- Regenerate the client from the repo root after backend route changes: `make generate-api`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never a `Claude-Session:` trailer.

## Review Focus

1. A numeric ID column (BBBP's `num`) shows as `12`, not `12.0`, and search still matches it. Pinned in Task 2.
2. Changing the identifier column changes the IDs on an already-trained protocol's scorecard without retraining. Pinned in Task 2.
3. A blank ID cell reads as no ID, not an empty string, everywhere. Pinned in Task 2.
4. `q` that is only whitespace is no filter, not a match on everything containing a space. Pinned in Task 2.
5. In the wizard, choosing the identifier column as the structure or target column clears the identifier. Pinned in Task 3.

---

### Task 1: Record the identifier column

**Files:**
- Modify: `backend/src/daikonstudio/domain/data/dataset.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/models.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/repository.py`
- Modify: `backend/src/daikonstudio/application/ports/dataset_repository.py`
- Create: `backend/alembic/versions/012_dataset_id_column.py`
- Modify: `backend/src/daikonstudio/application/auth.py`
- Create: `backend/src/daikonstudio/application/data/compound_ids.py`
- Create: `backend/src/daikonstudio/application/data/set_dataset_id_column.py`
- Modify: `backend/src/daikonstudio/application/data/create_dataset.py`
- Modify: `backend/src/daikonstudio/interface/routes/datasets.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Create: `backend/tests/api/test_dataset_id_column.py`

**Interfaces:**
- Produces: `check_id_column(columns, *, id_column, structure_column, target_column) -> None` (raises `ValidationError`) in `domain/data/dataset.py`; `Dataset.id_column: str | None`.
- Produces: `DatasetRepository.set_id_column(workspace_id, dataset_id, id_column: str | None) -> None`.
- Produces: `is_editor(auth) -> bool` in `application/auth.py`.
- Produces in `application/data/compound_ids.py`: `snapshot_columns(store, dataset) -> list[str]`, `eligible_id_columns(columns, dataset) -> list[str]`, `read_compound_ids(store, dataset) -> dict[str, str] | None`, `id_text(column) -> pl.Expr` (trimmed text, blank → null).
- Produces: `SetDatasetIdColumn`, `GetDatasetColumns` use cases; `PUT /api/v1/datasets/{id}/id-column`, `GET /api/v1/datasets/{id}/columns`; `DatasetResponse.id_column`, `DatasetResponse.can_edit`; `CreateDatasetBody.id_column`.
- Produces (tests): `_csv_with_ids()`, `_create(client, csv_upload, **overrides) -> httpx.Response` in `tests/api/test_dataset_id_column.py`.

- [ ] **Step 1: Write the failing API tests**

`backend/tests/api/test_dataset_id_column.py`:

```python
"""A dataset's identifier column: chosen at upload or later, shown wherever its compounds are."""

from tests.api import test_protocols

_STRUCTURES = test_protocols._STRUCTURES


def _csv_with_ids() -> bytes:
    rows = "\n".join(
        f"{smiles},{1.0 + 0.37 * index},cpd-{index},{index}"
        for index, smiles in enumerate(_STRUCTURES)
    )
    return f"smiles,y,name,num\n{rows}\n".encode()


async def _create(client, csv_upload, **overrides):
    upload_ref = await csv_upload(_csv_with_ids())
    body = {
        "name": "solubility",
        "upload_ref": upload_ref,
        "structure_column": "smiles",
        "target": {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
        "split": {"strategy": "random", "seed": 1},
    }
    body.update(overrides)
    return await client.post("/api/v1/datasets", json=body)


async def test_the_identifier_column_can_be_chosen_at_upload(client, csv_upload):
    response = await _create(client, csv_upload, id_column="name")

    assert response.status_code == 201, response.text
    assert response.json()["id_column"] == "name"


async def test_an_identifier_column_must_exist_and_not_be_reserved(client, csv_upload):
    missing = await _create(client, csv_upload, id_column="nope")
    assert missing.status_code == 422
    assert missing.json()["message"] == "Column 'nope' is not in the uploaded file."

    reserved = await _create(client, csv_upload, id_column="smiles")
    assert reserved.status_code == 422
    assert reserved.json()["message"] == (
        "Choose an identifier column other than the structure, target or split column."
    )


async def test_an_editor_sets_changes_and_clears_it_later(client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]
    url = f"/api/v1/datasets/{dataset_id}/id-column"

    assert (await client.get(f"/api/v1/datasets/{dataset_id}/columns")).json() == {
        "columns": ["name", "num"]
    }
    first = await client.put(url, json={"id_column": "name"})
    assert first.status_code == 200, first.text
    assert first.json()["id_column"] == "name"
    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).json()["id_column"] == "name"

    assert (await client.put(url, json={"id_column": "split"})).status_code == 422
    assert (await client.put(url, json={"id_column": None})).json()["id_column"] is None


async def test_a_viewer_cannot_change_it(client, viewer_client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]

    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).json()["can_edit"] is True
    viewer = await viewer_client.get(f"/api/v1/datasets/{dataset_id}")
    assert viewer.json()["can_edit"] is False
    refused = await viewer_client.put(
        f"/api/v1/datasets/{dataset_id}/id-column", json={"id_column": "name"}
    )
    assert refused.status_code == 403
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && env OMP_NUM_THREADS=1 uv run pytest tests/api/test_dataset_id_column.py -q`
Expected: 4 failed (`id_column` rejected by `extra="forbid"` with 422, `KeyError: 'id_column'`, 404/405 on the new routes).

- [ ] **Step 3: Domain, persistence and migration**

`domain/data/dataset.py`: import `ValidationError` alongside `ConflictError`; add the constructor keyword `id_column: str | None = None` after `created_by`, stored as `self.id_column = id_column` with the comment "# Which snapshot column holds the compounds' own IDs, if any. Display metadata: not part of `content_hash`, and changeable after freezing.", and add:

```python
def check_id_column(
    columns: Sequence[str], *, id_column: str, structure_column: str, target_column: str
) -> None:
    """An identifier is any stored column but the ones that already mean something."""
    if id_column in (structure_column, target_column, "split"):
        raise ValidationError(
            "Choose an identifier column other than the structure, target or split column."
        )
    if id_column not in columns:
        raise ValidationError(f"Column '{id_column}' is not in the uploaded file.")
```

(`from collections.abc import Sequence`.)

`DatasetModel`: `id_column: Mapped[str | None] = mapped_column(String(128), nullable=True)` with the comment "# The snapshot column holding compound IDs; display metadata, see Dataset.id_column." Both mappers carry `id_column`.

`application/ports/dataset_repository.py`: add `async def set_id_column(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID, id_column: str | None) -> None: ...` and extend the docstring: "`set_id_column` changes the one setting that is not frozen: which column holds IDs."

Repository (import `update as sa_update` and `func` from `sqlalchemy`):

```python
    async def set_id_column(
        self, workspace_id: uuid.UUID, dataset_id: uuid.UUID, id_column: str | None
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_update(DatasetModel)
                .where(DatasetModel.id == dataset_id, DatasetModel.workspace_id == workspace_id)
                .values(
                    id_column=id_column,
                    version=DatasetModel.version + 1,
                    updated_at=func.now(),
                )
            )
            await session.commit()
```

`backend/alembic/versions/012_dataset_id_column.py`:

```python
"""datasets.id_column

Which snapshot column holds the compounds' own IDs. Every snapshot already keeps
the upload's other columns; nothing recorded which one is the identifier. NULL
for every existing dataset, which can be given one later from its page.

Revision ID: 012
Revises: 011
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("id_column", sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "id_column")
```

- [ ] **Step 4: The snapshot helper and the use cases**

`application/auth.py`, after `require_admin`:

```python
def is_editor(auth: AuthContext | None) -> bool:
    """For the UI: whether this viewer may change a dataset's settings."""
    return auth is None or _ROLE_RANK.get(auth.workspace_role, -1) >= _ROLE_RANK["editor"]
```

`application/data/compound_ids.py`:

```python
"""A dataset's compound IDs, read from its snapshot when a page asks for them.

Never copied into scorecard inputs or map files, so naming or changing the
identifier column takes effect everywhere at once. Snapshot structures are
canonical and unique (duplicates are merged at upload), and the scorecard's and
the map's structures are the snapshot's own strings, so a structure is the key.
"""

from __future__ import annotations

import io

import polars as pl

from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.domain.data.dataset import Dataset


def id_text(column: str) -> pl.Expr:
    """IDs as trimmed text, blanks as null. An integer column prints as "12"; a
    text column keeps leading zeros ("007")."""
    text = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.when(text == "").then(None).otherwise(text)


def snapshot_columns(store: BlobStore, dataset: Dataset) -> list[str]:
    raw = store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
    return list(pl.read_parquet_schema(io.BytesIO(raw)).keys())


def eligible_id_columns(columns: list[str], dataset: Dataset) -> list[str]:
    reserved = {dataset.structure_column, dataset.target.column, "split"}
    return [column for column in columns if column not in reserved]


def read_compound_ids(store: BlobStore, dataset: Dataset) -> dict[str, str] | None:
    """Structure → ID, or None when the dataset names no identifier column."""
    if dataset.id_column is None:
        return None
    raw = store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
    frame = pl.read_parquet(
        io.BytesIO(raw), columns=[dataset.structure_column, dataset.id_column]
    ).select(
        pl.col(dataset.structure_column).cast(pl.String).alias("structure"),
        id_text(dataset.id_column).alias("id"),
    )
    return {
        structure: compound_id
        for structure, compound_id in frame.iter_rows()
        if compound_id is not None
    }
```

`application/data/set_dataset_id_column.py`:

```python
"""Set or clear a Dataset's identifier column, the one setting not frozen with it.

Display metadata only: no snapshot, score or content hash changes, and it can be
changed back, so any editor may set it."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.compound_ids import eligible_id_columns, snapshot_columns
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.data.dataset import Dataset, check_id_column
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


@dataclass(frozen=True, kw_only=True)
class SetDatasetIdColumnCommand:
    dataset_id: uuid.UUID
    id_column: str | None


@dataclass(frozen=True, kw_only=True)
class GetDatasetColumnsQuery:
    dataset_id: uuid.UUID


class SetDatasetIdColumn:
    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository = repository
        self._store = store

    async def __call__(
        self, command: SetDatasetIdColumnCommand, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._repository.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        if command.id_column is not None:
            try:
                columns = snapshot_columns(self._store, dataset)
            except FileNotFoundError:
                return Failure(NotFoundError("Stored dataset file", str(dataset.id)))
            try:
                check_id_column(
                    columns,
                    id_column=command.id_column,
                    structure_column=dataset.structure_column,
                    target_column=dataset.target.column,
                )
            except ValidationError as error:
                return Failure(error)
        await self._repository.set_id_column(auth.workspace_id, dataset.id, command.id_column)
        dataset.id_column = command.id_column
        return Success(dataset)


class GetDatasetColumns:
    """The snapshot columns eligible as an identifier, for the picker."""

    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository = repository
        self._store = store

    async def __call__(
        self, query: GetDatasetColumnsQuery, auth: AuthContext | None = None
    ) -> Result[list[str], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._repository.get(auth.workspace_id, query.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(query.dataset_id)))
        try:
            columns = snapshot_columns(self._store, dataset)
        except FileNotFoundError:
            return Failure(NotFoundError("Stored dataset file", str(dataset.id)))
        return Success(eligible_id_columns(columns, dataset))
```

`application/data/create_dataset.py`: `CreateDatasetCommand` gains `id_column: str | None = None`. Right after the `missing` columns check returns, add:

```python
        if command.id_column is not None:
            try:
                check_id_column(
                    frame.columns,
                    id_column=command.id_column,
                    structure_column=command.structure_column,
                    target_column=command.target.column,
                )
            except ValidationError as error:
                return Failure(error)
```

(import `check_id_column` from `domain.data.dataset`), and pass `id_column=command.id_column` to `Dataset(...)`.

- [ ] **Step 5: Routes and wiring**

`interface/routes/datasets.py`:
- Docstring: replace "There is no PATCH: a Dataset is immutable and cited by id." with "There is no PATCH: a Dataset is immutable and cited by id. The one setting that is not frozen, its identifier column, has its own `PUT .../id-column`."
- `CreateDatasetBody` gains `id_column: str | None = Field(default=None, max_length=128)`; the route passes `id_column=body.id_column`.
- `DatasetResponse` gains, after `can_delete`:

```python
    # Which snapshot column holds the compounds' own IDs, if any.
    id_column: str | None
    # Whether this viewer may change its settings (the identifier column).
    can_edit: bool
```

  with `id_column=dataset.id_column, can_edit=is_editor(auth),` in `from_domain` (import `is_editor`).
- New models and routes (import the use cases and commands from `application.data.set_dataset_id_column`):

```python
class SetIdColumnBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_column: str | None = Field(max_length=128)


class DatasetColumnsResponse(BaseModel):
    """Snapshot columns that may be named as the identifier."""

    columns: list[str]


@router.put("/{dataset_id}/id-column", response_model=DatasetResponse)
async def set_dataset_id_column(
    dataset_id: uuid.UUID, body: SetIdColumnBody, auth: AuthDep, service: SetDatasetIdColumnDep
) -> DatasetResponse:
    """Any editor; `null` clears it. Display metadata: nothing frozen changes."""
    dataset = result_to_response(
        await service(
            SetDatasetIdColumnCommand(dataset_id=dataset_id, id_column=body.id_column), auth=auth
        )
    )
    return DatasetResponse.from_domain(dataset, auth=auth)


@router.get("/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_dataset_columns(
    dataset_id: uuid.UUID, auth: AuthDep, service: GetDatasetColumnsDep
) -> DatasetColumnsResponse:
    columns = result_to_response(
        await service(GetDatasetColumnsQuery(dataset_id=dataset_id), auth=auth)
    )
    return DatasetColumnsResponse(columns=columns)
```

  with `SetDatasetIdColumnDep` / `GetDatasetColumnsDep` aliases beside `DeleteDatasetDep`.
- `infrastructure/di/container.py`, beside `DeleteDataset`:

```python
    container.define(
        SetDatasetIdColumn, lambda c: SetDatasetIdColumn(_datasets(c), c[BlobStore])
    )
    container.define(GetDatasetColumns, lambda c: GetDatasetColumns(_datasets(c), c[BlobStore]))
```

- [ ] **Step 6: Run the tests and the gates; commit**

Run: `cd backend && env OMP_NUM_THREADS=1 uv run pytest tests/api/test_dataset_id_column.py tests/integration/test_migrations.py -q`
Expected: all pass. Then the five backend gates.

```bash
git add backend
git commit -m "feat(datasets): record which column holds compound IDs"
```

---

### Task 2: Show IDs where compounds are read

**Files:**
- Modify: `backend/src/daikonstudio/application/data/get_dataset_compounds.py`
- Modify: `backend/src/daikonstudio/domain/execution/scorecard.py` (`WorstRow`)
- Modify: `backend/src/daikonstudio/application/catalog/get_scorecard.py`
- Modify: `backend/src/daikonstudio/application/catalog/get_chemical_space.py` (`MapCompound`, `GetProtocolChemicalSpaceCompounds`)
- Modify: `backend/src/daikonstudio/interface/routes/datasets.py` (`CompoundResponse`, compounds route `q`)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py` (`WorstRowResponse`, `MapCompoundResponse`)
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Modify: `backend/tests/unit/catalog/test_get_scorecard.py`
- Modify: `backend/tests/api/test_dataset_id_column.py`

**Interfaces:**
- Consumes: `read_compound_ids`, `id_text` (Task 1); `_create`, `_csv_with_ids` (Task 1).
- Produces: `Compound.compound_id`, `GetDatasetCompoundsQuery.q`, `WorstRow.compound_id: str | None = None`, `MapCompound.compound_id: str | None = None`; `GetScorecard(protocols, store, normalizer, datasets)`; `GetProtocolChemicalSpaceCompounds(protocols, store, datasets)`; API `compound_id` on compound rows, worst rows and map compounds; `q` on `GET /datasets/{id}/compounds`.

- [ ] **Step 1: Write the failing API tests**

Append to `backend/tests/api/test_dataset_id_column.py`:

```python
async def _compounds(client, dataset_id: str, **params):
    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds", params={"limit": 200, **params}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_compounds_carry_their_id_and_can_be_searched(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="name")).json()["id"]

    every = await _compounds(client, dataset_id)
    assert sorted(item["compound_id"] for item in every["items"]) == sorted(
        f"cpd-{index}" for index in range(len(_STRUCTURES))
    )

    found = await _compounds(client, dataset_id, q="  CPD-1 ")
    assert found["total"] == 11  # cpd-1 and cpd-10 to cpd-19
    assert all("cpd-1" in item["compound_id"] for item in found["items"])

    # Whitespace alone is no filter.
    assert (await _compounds(client, dataset_id, q="   "))["total"] == len(_STRUCTURES)


async def test_a_numeric_id_column_reads_as_integers(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="num")).json()["id"]

    ids = {item["compound_id"] for item in (await _compounds(client, dataset_id))["items"]}
    assert ids == {str(index) for index in range(len(_STRUCTURES))}
    assert (await _compounds(client, dataset_id, q="12"))["total"] == 1


async def test_search_needs_an_identifier_column(client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]

    refused = await client.get(f"/api/v1/datasets/{dataset_id}/compounds", params={"q": "cpd"})
    assert refused.status_code == 422
    assert refused.json()["message"] == "This dataset has no identifier column."
    assert all(item["compound_id"] is None for item in (await _compounds(client, dataset_id))["items"])


async def test_a_protocols_errors_and_map_show_ids_and_follow_a_change(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="name")).json()["id"]
    by_structure = {
        item["structure"]: item["compound_id"]
        for item in (await _compounds(client, dataset_id))["items"]
    }
    train = await test_protocols._train(client, dataset_id)
    assert train.status_code == 202, train.text
    run = (await client.get(f"/api/v1/runs/{train.json()['id']}")).json()
    protocol_id = run["protocol_id"]

    scorecard = (await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")).json()
    assert scorecard["worst_rows"]
    for row in scorecard["worst_rows"]:
        assert row["compound_id"] == by_structure[row["structure"]]

    compounds = (
        await client.get(
            f"/api/v1/protocols/{protocol_id}/chemical-space/compounds",
            params={"indices": [0, 1]},
        )
    ).json()
    assert [item["compound_id"] for item in compounds] == [
        by_structure[item["structure"]] for item in compounds
    ]

    # Switching the column changes the IDs on the trained protocol at once.
    await client.put(f"/api/v1/datasets/{dataset_id}/id-column", json={"id_column": "num"})
    switched = (await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")).json()
    assert all(row["compound_id"].isdigit() for row in switched["worst_rows"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && env OMP_NUM_THREADS=1 uv run pytest tests/api/test_dataset_id_column.py -q -k "compounds or numeric or search or errors"`
Expected: 4 failed (`KeyError: 'compound_id'`, `q` ignored so totals are 20, no 422).

- [ ] **Step 3: Compounds endpoint**

`get_dataset_compounds.py`: `Compound` gains `compound_id: str | None`; `GetDatasetCompoundsQuery` gains `q: str | None = None`. After loading the dataset:

```python
        search = (query.q or "").strip().lower()
        if search and dataset.id_column is None:
            return Failure(ValidationError("This dataset has no identifier column."))
```

(import `ValidationError` and `from daikonstudio.application.data.compound_ids import id_text`). The `select` adds

```python
            (
                id_text(dataset.id_column)
                if dataset.id_column is not None
                else pl.lit(None, dtype=pl.String)
            ).alias("compound_id"),
```

and after the split filter:

```python
        if search:
            frame = frame.filter(
                pl.col("compound_id").str.to_lowercase().str.contains(search, literal=True)
            )
```

Each `Compound(...)` gains `compound_id=row["compound_id"]`. Update the module docstring's "Deliberately narrow" sentence to "Deliberately narrow: structure, target, partition and, when the dataset names one, the compound's ID, searchable by ID."

Route: `CompoundResponse` gains `compound_id: str | None` (set in `from_domain`); the route gains `q: Annotated[str | None, Query(max_length=128)] = None` passed into the query (import `Query` from `fastapi`).

- [ ] **Step 4: Scorecard worst rows**

`WorstRow` gains, as its last field, `compound_id: str | None = None` with the docstring line "`compound_id` is the compound's own ID when its dataset names an identifier column, looked up when the scorecard is read."

`get_scorecard.py`: the constructor gains `datasets: DatasetRepository` as the last parameter. After `build_scorecard` returns (bind it to `scorecard` instead of returning it inline):

```python
        dataset = await self._datasets.get(protocol.workspace_id, protocol.dataset_id)
        ids = None
        if dataset is not None:
            try:
                ids = await asyncio.to_thread(read_compound_ids, self._store, dataset)
            except FileNotFoundError:
                ids = None
        if ids:
            scorecard = replace(
                scorecard,
                worst_rows=[
                    replace(row, compound_id=ids.get(row.structure))
                    for row in scorecard.worst_rows
                ],
            )
        return Success(scorecard)
```

(`from dataclasses import dataclass, replace`.) `WorstRowResponse` gains `compound_id: str | None`. DI: `GetScorecard(_protocols(c), c[BlobStore], c[StructureNormalizer], _datasets(c))`. In `tests/unit/catalog/test_get_scorecard.py` pass `datasets=_NoDatasets()` where

```python
class _NoDatasets:
    async def get(self, workspace_id, dataset_id):
        return None
```

- [ ] **Step 5: Map compounds**

`MapCompound` gains `compound_id: str | None = None`. `GetProtocolChemicalSpaceCompounds.__init__` gains `datasets: DatasetRepository`; after building the list:

```python
        found = [...]  # the existing comprehension, bound to a name
        dataset = await self._datasets.get(protocol.workspace_id, protocol.dataset_id)
        if found and dataset is not None:
            try:
                ids = await asyncio.to_thread(read_compound_ids, self._store, dataset)
            except FileNotFoundError:
                ids = None
            if ids:
                found = [replace(item, compound_id=ids.get(item.structure)) for item in found]
        return Success(found)
```

`MapCompoundResponse` gains `compound_id: str | None` and the route passes `compound_id=i.compound_id`. DI: `GetProtocolChemicalSpaceCompounds(_protocols(c), c[BlobStore], _datasets(c))`.

- [ ] **Step 6: Run the tests and the gates; commit**

Run: `cd backend && env OMP_NUM_THREADS=1 uv run pytest tests/api/test_dataset_id_column.py tests/unit/catalog/test_get_scorecard.py tests/api/test_chemical_space.py -q`
Expected: all pass. Then the five backend gates.

```bash
git add backend
git commit -m "feat(datasets): show each training compound's ID, and search by it"
```

---

### Task 3: Choose the identifier column in the UI

**Files:**
- Regenerate: `frontend/openapi.json`, `frontend/src/shared/lib/api/**` (`make generate-api`)
- Move: `frontend/src/features/runs/lib/guess-id-column.ts` (+ `.test.ts`) → `frontend/src/shared/lib/guess-id-column.ts` (+ `.test.ts`)
- Modify: `frontend/src/features/runs/components/predict-wizard.tsx` (import path)
- Create: `frontend/src/features/datasets/lib/draft-from-upload.ts` (+ `.test.ts`)
- Modify: `frontend/src/features/datasets/types/index.ts` (`DatasetDraft.idColumn`, `EMPTY_DRAFT`)
- Modify: `frontend/src/features/datasets/components/dataset-wizard.tsx`
- Modify: `frontend/src/features/datasets/hooks/use-datasets.ts`
- Create: `frontend/src/features/datasets/components/id-column-field.tsx` (+ `.test.tsx`)
- Modify: `frontend/src/features/datasets/components/dataset-detail.tsx`

**Interfaces:**
- Consumes: `PUT /datasets/{id}/id-column`, `GET /datasets/{id}/columns`, `DatasetResponse.id_column`, `.can_edit`, `CreateDatasetBody.id_column` (Task 1).
- Produces: `guessIdColumn` at `@/shared/lib/guess-id-column`; `draftFromUpload(columns, rows, fileName)`; `useDatasetColumns(id, enabled)`, `useSetDatasetIdColumn()`; `IdColumnField({ dataset })`.

- [ ] **Step 1: Regenerate the client and move the guess**

Run `make generate-api` from the repo root. `git mv` the guess and its test to `frontend/src/shared/lib/`; update the predict wizard's import to `@/shared/lib/guess-id-column`; the moved test's import stays `./guess-id-column`.

- [ ] **Step 2: Write the failing tests**

`frontend/src/features/datasets/lib/draft-from-upload.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { draftFromUpload, withColumns } from "./draft-from-upload";

describe("draftFromUpload", () => {
  it("guesses structure, target and identifier columns from the headers", () => {
    const draft = draftFromUpload(["Structure", "pIC50", "NATHAN ID"], [], "screen.csv");
    expect(draft).toMatchObject({
      name: "screen",
      structureColumn: "Structure",
      targetColumn: "pIC50",
      idColumn: "NATHAN ID",
    });
  });

  it("never guesses the target column as the identifier", () => {
    const draft = draftFromUpload(["smiles", "y", "compound_id"], [], "a.csv");
    expect(draft).toMatchObject({ targetColumn: "y", idColumn: "compound_id" });
  });
});

describe("withColumns", () => {
  it("clears the identifier when it becomes the structure or target column", () => {
    const draft = draftFromUpload(["smiles", "y", "name"], [], "a.csv");
    expect(withColumns(draft, { targetColumn: "name" }).idColumn).toBeNull();
    expect(withColumns(draft, { structureColumn: "name" }).idColumn).toBeNull();
    expect(withColumns(draft, { targetColumn: "y" }).idColumn).toBe("name");
  });
});
```

`frontend/src/features/datasets/components/id-column-field.test.tsx`:

```tsx
import type { DatasetResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { IdColumnField } from "./id-column-field";

vi.mock("../hooks/use-datasets", () => ({
  useDatasetColumns: () => ({ data: { columns: ["name", "num"] } }),
  useSetDatasetIdColumn: () => ({ mutate: vi.fn(), isPending: false }),
}));

const dataset = (overrides: Partial<DatasetResponse>) =>
  ({ id: "d-1", id_column: "name", can_edit: true, ...overrides }) as DatasetResponse;

describe("IdColumnField", () => {
  it("lets an editor change it", () => {
    render(<IdColumnField dataset={dataset({})} />);
    expect(screen.getByRole("combobox", { name: "Identifier column" })).toHaveTextContent("name");
  });

  it("shows a viewer the value only", () => {
    render(<IdColumnField dataset={dataset({ can_edit: false })} />);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByText("name")).toBeInTheDocument();
  });

  it("says when there is none", () => {
    render(<IdColumnField dataset={dataset({ can_edit: false, id_column: null })} />);
    expect(screen.getByText("None")).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cd frontend && pnpm test src/features/datasets/lib/draft-from-upload src/features/datasets/components/id-column-field`
Expected: FAIL, both imports unresolved.

- [ ] **Step 4: Implement**

`DatasetDraft` gains `idColumn: string | null`; `EMPTY_DRAFT.idColumn = null`.

`frontend/src/features/datasets/lib/draft-from-upload.ts`:

```ts
import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { EMPTY_DRAFT, type DatasetDraft } from "../types";
import { guessStructureColumn, looksBinary } from "./parse-csv";

/** The wizard's starting draft for an upload: every column guessed from its headers. */
export function draftFromUpload(
  columns: string[],
  rows: Record<string, string>[],
  fileName: string,
): DatasetDraft {
  const structureColumn = guessStructureColumn(columns);
  const targetColumn = columns.find((column) => column !== structureColumn) ?? "";
  return {
    ...EMPTY_DRAFT,
    name: fileName.replace(/\.csv$/i, ""),
    structureColumn,
    targetColumn,
    kind: targetColumn && looksBinary(rows, targetColumn) ? "binary" : "numeric",
    idColumn: guessIdColumn(
      columns.filter((column) => column !== targetColumn),
      structureColumn,
    ),
  };
}

/** A column change that keeps the identifier distinct from the structure and target. */
export function withColumns(
  draft: DatasetDraft,
  changes: Partial<Pick<DatasetDraft, "structureColumn" | "targetColumn">>,
): DatasetDraft {
  const next = { ...draft, ...changes };
  const clash = next.idColumn === next.structureColumn || next.idColumn === next.targetColumn;
  return clash ? { ...next, idColumn: null } : next;
}
```

(Check the wizard's existing imports for where `guessStructureColumn`/`looksBinary` live and import from there; the wizard keeps `file` and any other fields it sets today — merge `draftFromUpload(...)` into its existing `setDraft` call rather than dropping them.)

Wizard Columns step: switch the column grid to `sm:grid-cols-3`; route the Structures and Value-to-predict `onValueChange` handlers through `setDraft((prev) => withColumns(prev, {...}))` (keeping the target's `kind` update); add a third field:

```tsx
                <div className="space-y-1.5">
                  <Label>Identifier (optional)</Label>
                  <Select
                    value={draft.idColumn ?? NO_ID}
                    onValueChange={(value) => patch({ idColumn: value === NO_ID ? null : value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NO_ID}>None</SelectItem>
                      {preview.columns
                        .filter(
                          (column) =>
                            column !== draft.structureColumn && column !== draft.targetColumn,
                        )
                        .map((column) => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>
```

with `const NO_ID = "__none__";` at module level, and `id_column: draft.idColumn` in the create body.

Hooks in `use-datasets.ts` (import `DatasetColumnsResponse`, `DatasetResponse` types as generated):

```ts
/** Snapshot columns that may be named as the identifier. */
export function useDatasetColumns(id: string, enabled: boolean) {
  return useQuery({
    queryKey: [...DATASET_KEY, id, "columns"],
    queryFn: () =>
      customInstance<DatasetColumnsResponse>({
        url: `${API_V1}/datasets/${id}/columns`,
        method: "GET",
      }),
    enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useSetDatasetIdColumn() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, idColumn }: { id: string; idColumn: string | null }) =>
      customInstance<DatasetResponse>({
        url: `${API_V1}/datasets/${id}/id-column`,
        method: "PUT",
        data: { id_column: idColumn },
      }),
    onSuccess: (dataset) => {
      queryClient.setQueryData([...DATASET_KEY, dataset.id], dataset);
      queryClient.invalidateQueries({ queryKey: [...DATASET_COMPOUNDS_KEY, dataset.id] });
      // Scorecards and map tooltips of its protocols show these IDs too: mark
      // everything stale, refetching nothing now.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}
```

`frontend/src/features/datasets/components/id-column-field.tsx`:

```tsx
"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import type { DatasetResponse } from "@/shared/lib/api/model";
import { useState } from "react";
import { useDatasetColumns, useSetDatasetIdColumn } from "../hooks/use-datasets";

const NONE = "__none__";

/** Which column holds the compounds' own IDs; any editor may change it. */
export function IdColumnField({ dataset }: { dataset: DatasetResponse }) {
  const [open, setOpen] = useState(false);
  const columns = useDatasetColumns(dataset.id, dataset.can_edit && open);
  const save = useSetDatasetIdColumn();

  if (!dataset.can_edit) {
    return <span className="font-mono">{dataset.id_column ?? "None"}</span>;
  }
  const options = columns.data?.columns ?? (dataset.id_column ? [dataset.id_column] : []);
  return (
    <Select
      value={dataset.id_column ?? NONE}
      onOpenChange={setOpen}
      disabled={save.isPending}
      onValueChange={(value) =>
        save.mutate({ id: dataset.id, idColumn: value === NONE ? null : value })
      }
    >
      <SelectTrigger aria-label="Identifier column" className="h-8 w-48 font-mono text-xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={NONE}>None</SelectItem>
        {options.map((column) => (
          <SelectItem key={column} value={column} className="font-mono">
            {column}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
```

`dataset-detail.tsx` Overview "What this predicts": add a fifth field, and widen the grid to `sm:grid-cols-5`:

```tsx
                <Field label="Identifier column" value={<IdColumnField dataset={dataset} />} />
```

- [ ] **Step 5: Run the tests and the gates; commit**

Run: `cd frontend && pnpm test src/features/datasets src/shared/lib src/features/runs`
Expected: all pass. Then the three frontend gates.

```bash
git add frontend
git commit -m "feat(datasets): choose the identifier column at upload or on the dataset page"
```

---

### Task 4: Show IDs in the UI, and search by ID

**Files:**
- Modify: `frontend/src/features/datasets/hooks/use-datasets.ts` (`CompoundQuery.q`)
- Modify: `frontend/src/features/datasets/components/compound-browser.tsx`
- Create: `frontend/src/features/datasets/components/compound-browser.test.tsx`
- Modify: `frontend/src/features/protocols/components/largest-errors.tsx` (+ its test)
- Modify: `frontend/src/features/protocols/components/protocol-chemical-space.tsx` (`MapCompoundTooltip`)

**Interfaces:**
- Consumes: `compound_id` on compounds, worst rows and map compounds; `q` (Task 2).

- [ ] **Step 1: Write the failing tests**

`frontend/src/features/datasets/components/compound-browser.test.tsx`:

```tsx
import type { DatasetResponse } from "@/shared/lib/api/model";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CompoundBrowser } from "./compound-browser";

const hoisted = vi.hoisted(() => ({ queries: [] as Record<string, unknown>[] }));

vi.mock("@/shared/components/chemistry/structure-thumbnail", () => ({
  StructureThumbnail: () => null,
}));
vi.mock("../hooks/use-datasets", () => ({
  useDatasetCompounds: (_id: string, query: Record<string, unknown>) => {
    hoisted.queries.push(query);
    return {
      isLoading: false,
      isError: false,
      data: { items: [{ structure: "CCO", target: 1, split: "train", compound_id: "RU-7" }], total: 1 },
    };
  },
}));

const dataset = (idColumn: string | null) =>
  ({
    id: "d-1",
    id_column: idColumn,
    target: { column: "y", unit: null },
  }) as unknown as DatasetResponse;

describe("CompoundBrowser", () => {
  beforeEach(() => {
    hoisted.queries = [];
  });

  it("shows each compound's ID and searches by it", async () => {
    vi.useFakeTimers();
    render(<CompoundBrowser dataset={dataset("RU ID")} />);

    expect(screen.getByText("RU-7")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("searchbox", { name: "Search by ID" }), {
      target: { value: "RU-7" },
    });
    await act(async () => vi.advanceTimersByTime(400));

    expect(hoisted.queries.at(-1)).toMatchObject({ q: "RU-7", offset: 0 });
    vi.useRealTimers();
  });

  it("offers no search without an identifier column", () => {
    render(<CompoundBrowser dataset={dataset(null)} />);
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
  });
});
```

In `largest-errors.test.tsx`, add a `compound_id` argument to `row(...)` (default `null`) and a test:

```tsx
  it("shows a compound's own ID on its card", () => {
    render(<LargestErrors scorecard={scorecard([{ ...row("a", "c1ccccc1", 0.5), compound_id: "RU-7" }])} />);
    expect(screen.getByText("RU-7")).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd frontend && pnpm test src/features/datasets/components/compound-browser src/features/protocols/components/largest-errors`
Expected: FAIL (no "RU-7" text, no search box).

- [ ] **Step 3: Implement**

`CompoundQuery` gains `q?: string`.

`compound-browser.tsx`: add search state with a 300 ms debounce, reset to the first page on change, only when `dataset.id_column`:

```tsx
  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQ(search.trim());
      setOffset(0);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [search]);
```

pass `q: q || undefined` into `useDatasetCompounds`, render above the table when `dataset.id_column`:

```tsx
      {dataset.id_column && (
        <Input
          type="search"
          aria-label="Search by ID"
          placeholder={`Search by ${dataset.id_column}`}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="h-8 max-w-xs"
        />
      )}
```

and, when `dataset.id_column`, an ID column: `<TableHead>{dataset.id_column}</TableHead>` after Structure, and `<TableCell className="font-mono text-xs">{compound.compound_id ?? "—"}</TableCell>`; the skeleton row's `colSpan` follows the column count.

`largest-errors.tsx`: above `<StructureThumbnail ... />`:

```tsx
              {row.compound_id && (
                <p className="w-full truncate text-center font-mono text-xs font-medium" title={row.compound_id}>
                  {row.compound_id}
                </p>
              )}
```

`MapCompoundTooltip`: before the partition line, `{data.compound_id && <p className="font-mono font-medium">{data.compound_id}</p>}`.

- [ ] **Step 4: Run the tests and the gates; commit**

Run: `cd frontend && pnpm test src/features`
Expected: all pass. Then the three frontend gates.

```bash
git add frontend
git commit -m "feat(datasets): show compound IDs on the Compounds tab, error cards and map"
```

---

## After the tasks

- Final whole-branch review on the most capable model, with this plan's Review Focus.
- Apply migration 012 to the dev database (`make migrate`).
- Live check in Chrome: set `NATHAN ID` on the 400k dataset and search one ID; set `name` on BBBP and see IDs on its Compounds tab and on one of its protocols' error cards and map tooltips.
