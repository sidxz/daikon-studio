# Delete Datasets and Draft Protocols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin, or the person who created it, can permanently delete a dataset nothing depends on, or a draft protocol, from its detail page.

**Architecture:** A nullable `created_by` on datasets and protocols (migration 011, protocols backfilled from their training run) feeds one policy function, `may_delete`, used both by the two DELETE use cases and by a `can_delete` flag on the responses. Each use case checks dependents, deletes rows (runs first, then the item), then deletes the item's blob folder by prefix, logging rather than failing if the folder delete fails. The frontend adds a Delete button with a confirm dialog on each detail page.

**Tech Stack:** FastAPI, SQLAlchemy async, Alembic, fsspec, pytest with testcontainers Postgres; Next 16, React 19, TanStack Query v5, Radix AlertDialog, vitest.

**Spec:** `docs/superpowers/specs/2026-10-03-delete-datasets-and-draft-protocols-design.md`

## Global Constraints

- Who may delete: workspace role admin or owner; or the creator while their role is editor or higher. `created_by` NULL means admin-only.
- Only draft protocols are deletable. Published: 409 "Published protocols cannot be deleted."
- Dataset with any protocol: 409 "Protocols trained on this dataset must be deleted first. A dataset used by a published protocol cannot be deleted."
- Dataset with a pending or running training run: 409 "A training run on this dataset is still in progress."
- Draft whose training run is pending or running: 409 "This protocol's training run is still finishing. Try again when it completes."
- Not permitted: 403 "Only an admin or the person who created it can delete this."
- Rows before files. Runs before the item they depend on. A failed folder delete is logged, never raised.
- Delete blobs by folder prefix ending in `/` only. Never delete `artifact_uri` directly: a versioned protocol's can point into another protocol's folder.
- UI copy is American English, academic and plain: no em-dash clause chains, en dash only in ranges and "Bemis–Murcko".
- Backend gates, run from `backend/`: `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`, `uv run lint-imports`, `env OMP_NUM_THREADS=1 uv run pytest -q`.
- Frontend gates, run from `frontend/`: `pnpm lint`, `pnpm exec tsc --noEmit`, `pnpm test`.
- After changing backend routes or response models, regenerate the client from the repo root: `make generate-api`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never add a `Claude-Session:` trailer.

## Review Focus

1. Deleting `ws/protocols/abc/` must not delete a sibling folder that shares the prefix, such as `ws/protocols/abcdef/`. Pinned in Task 2, Step 1.
2. Deleting the same item twice answers 404 the second time, never 500. Pinned in Tasks 2 and 3.
3. A draft whose training run is still in its last phase ("Mapping chemical space", status running) is refused with 409. Pinned in Task 2.
4. A dataset deleted while its profile is computing stays deleted: the background task does not recreate its folder. Pinned in Task 3.
5. After deleting a dataset, uploading the same file again succeeds (the content-hash unique index no longer holds the old row). Pinned in Task 3.

Not covered by any test, for the final reviewer: `delete_prefix` on S3 (`s3fs.rm(path, recursive=True)` on a key prefix) is only exercised against the local filesystem.

---

### Task 1: Record creators and expose `can_delete`

**Files:**
- Modify: `backend/src/daikonstudio/application/auth.py`
- Modify: `backend/src/daikonstudio/domain/data/dataset.py` (constructor)
- Modify: `backend/src/daikonstudio/domain/catalog/protocol.py` (constructor)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/models.py` (`DatasetModel`)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/models.py` (`InSilicoProtocolModel`)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/repository.py` (`_to_domain`, `_to_model`)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/repository.py` (`_to_domain`, `_to_model`)
- Create: `backend/alembic/versions/011_created_by.py`
- Modify: `backend/src/daikonstudio/application/data/create_dataset.py:192-203`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:594-608`
- Modify: `backend/src/daikonstudio/interface/routes/datasets.py` (`DatasetResponse`, three callers)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py` (`ProtocolResponse`, two callers)
- Modify: `backend/tests/api/conftest.py` (new `other_editor_client` fixture)
- Create: `backend/tests/unit/test_may_delete.py`
- Create: `backend/tests/api/test_deleting.py`
- Modify: `backend/tests/integration/test_migrations.py`

**Interfaces:**
- Produces: `may_delete(auth: AuthContext | None, created_by: uuid.UUID | None) -> bool` and `require_may_delete(auth, created_by) -> None` (raises `AuthorizationError`) in `daikonstudio.application.auth`.
- Produces: `Dataset.created_by: uuid.UUID | None` and `InSilicoProtocol.created_by: uuid.UUID | None`, constructor keyword `created_by=None`.
- Produces: `can_delete: bool` on `DatasetResponse` and `ProtocolResponse`; `from_domain(..., *, auth: AuthContext | None)`.
- Produces (tests): fixture `other_editor_client`; helper `_train_protocol(client, dataset_id) -> tuple[str, str]` returning `(run_id, protocol_id)` in `tests/api/test_deleting.py`.

- [ ] **Step 1: Write the failing unit test for the policy**

`backend/tests/unit/test_may_delete.py`:

```python
"""Who may delete a dataset or a draft protocol."""

import uuid
from dataclasses import dataclass

import pytest

from daikonstudio.application.auth import may_delete, require_may_delete
from daikonstudio.domain.shared.errors import AuthorizationError

CREATOR = uuid.uuid4()


@dataclass(frozen=True)
class _Auth:
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_role: str


def _auth(role: str, user_id: uuid.UUID | None = None) -> _Auth:
    return _Auth(user_id=user_id or uuid.uuid4(), workspace_id=uuid.uuid4(), workspace_role=role)


@pytest.mark.parametrize(
    ("auth", "allowed"),
    [
        (_auth("editor", CREATOR), True),
        (_auth("editor"), False),
        (_auth("viewer", CREATOR), False),
        (_auth("admin"), True),
        (_auth("owner"), True),
    ],
)
def test_admins_and_the_creator_may_delete(auth, allowed):
    assert may_delete(auth, CREATOR) is allowed


def test_an_item_with_no_recorded_creator_is_admin_only():
    assert may_delete(_auth("editor"), None) is False
    assert may_delete(_auth("admin"), None) is True


def test_refusal_names_who_may_delete():
    with pytest.raises(AuthorizationError, match="Only an admin or the person who created it"):
        require_may_delete(_auth("editor"), CREATOR)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_may_delete.py -q`
Expected: FAIL with `ImportError: cannot import name 'may_delete'`.

- [ ] **Step 3: Implement the policy**

Append to `backend/src/daikonstudio/application/auth.py`:

```python
def may_delete(auth: AuthContext | None, created_by: uuid.UUID | None) -> bool:
    """Admins and owners may delete anything; the creator may delete their own
    while they still hold editor or higher. `created_by` is None for items made
    before creators were recorded (migration 011), so those are admin-only."""
    if auth is None:  # system/worker call
        return True
    rank = _ROLE_RANK.get(auth.workspace_role, -1)
    if rank >= _ROLE_RANK["admin"]:
        return True
    return created_by is not None and created_by == auth.user_id and rank >= _ROLE_RANK["editor"]


def require_may_delete(auth: AuthContext | None, created_by: uuid.UUID | None) -> None:
    if not may_delete(auth, created_by):
        raise AuthorizationError("Only an admin or the person who created it can delete this.")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_may_delete.py -q`
Expected: 7 passed.

- [ ] **Step 5: Write the failing API tests for `can_delete`**

Add to `backend/tests/api/conftest.py`, after `admin_client`:

```python
@pytest_asyncio.fixture
async def other_editor_client(app, signing_key, workspace_id) -> AsyncIterator[httpx.AsyncClient]:
    """A second editor in the same workspace: may read everything, delete only their own."""
    headers = auth_headers(signing_key[0], workspace_id=workspace_id, role="editor")
    async with _client(app, headers) as http_client:
        yield http_client
```

Create `backend/tests/api/test_deleting.py`:

```python
"""Deleting datasets and draft protocols: who may, and what goes with them."""

from tests.api import test_protocols

# A 20-compound dataset with a random split, and a training request, shared
# with the protocol suite.
_create_dataset = test_protocols._create_dataset
_train = test_protocols._train


async def _train_protocol(client, dataset_id: str) -> tuple[str, str]:
    """Train inline (tests use the in-process enqueuer); return (run_id, protocol_id)."""
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert run["status"] == "ready", run
    return run_id, run["protocol_id"]


async def test_only_the_creator_and_admins_may_delete_a_dataset(
    client, other_editor_client, admin_client, viewer_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    url = f"/api/v1/datasets/{dataset_id}"

    assert (await client.get(url)).json()["can_delete"] is True
    assert (await admin_client.get(url)).json()["can_delete"] is True
    assert (await other_editor_client.get(url)).json()["can_delete"] is False
    assert (await viewer_client.get(url)).json()["can_delete"] is False


async def test_a_draft_is_deletable_by_its_creator_until_it_is_published(
    client, other_editor_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    url = f"/api/v1/protocols/{protocol_id}"

    assert (await client.get(url)).json()["can_delete"] is True
    assert (await admin_client.get(url)).json()["can_delete"] is True
    assert (await other_editor_client.get(url)).json()["can_delete"] is False

    assert (await client.post(f"{url}/publish")).status_code == 204
    assert (await client.get(url)).json()["can_delete"] is False
    assert (await admin_client.get(url)).json()["can_delete"] is False
```

- [ ] **Step 6: Run them to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_deleting.py -q`
Expected: 2 failed with `KeyError: 'can_delete'`.

- [ ] **Step 7: Add `created_by` to the domain, the tables and the mappers**

In `domain/data/dataset.py`, add the keyword after `validation_report` and store it:

```python
        validation_report: ValidationReport,
        created_by: uuid.UUID | None = None,
        id: uuid.UUID | None = None,
```

```python
        self.validation_report = validation_report
        # Who created it, for the delete permission. None for datasets made before
        # migration 011 recorded it: those are admin-only.
        self.created_by = created_by
```

In `domain/catalog/protocol.py`, the same after `protocol_version`:

```python
        protocol_version: int = 1,
        created_by: uuid.UUID | None = None,
        id: uuid.UUID | None = None,
```

```python
        self.protocol_version = protocol_version
        # Who trained it, for the delete permission (backfilled by migration 011
        # from the training run's `requested_by`).
        self.created_by = created_by
```

In `DatasetModel` (after `validation_report`) and `InSilicoProtocolModel` (after `protocol_version`):

```python
    # Who created it, for the delete permission. NULL for rows made before 011.
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
```

In both repositories' `_to_domain`, add `created_by=model.created_by,`; in both `_to_model`, add `created_by=<dataset|protocol>.created_by,`.

- [ ] **Step 8: Write migration 011**

`backend/alembic/versions/011_created_by.py`:

```python
"""datasets.created_by and protocols.created_by

Who may delete a dataset or a draft protocol: an admin, or the person who
created it. Nothing recorded the creator before this revision.

Protocols are backfilled from the training run that produced them, whose
`requested_by` is exactly that person. Datasets cannot be: nothing recorded who
uploaded one, so existing datasets keep NULL and only an admin can delete them.

Revision ID: 011
Revises: 010
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | None = None
depends_on: str | None = None

# The earliest training run linked to each protocol. A protocol has one today;
# DISTINCT ON keeps this correct if that ever changes.
BACKFILL_PROTOCOL_CREATORS = """
UPDATE protocols AS p
SET created_by = r.requested_by
FROM (
    SELECT DISTINCT ON (protocol_id) protocol_id, workspace_id, requested_by
    FROM runs
    WHERE kind = 'training' AND protocol_id IS NOT NULL
    ORDER BY protocol_id, created_at
) AS r
WHERE r.protocol_id = p.id AND r.workspace_id = p.workspace_id AND p.created_by IS NULL
"""


def upgrade() -> None:
    op.add_column("datasets", sa.Column("created_by", sa.Uuid(), nullable=True))
    op.add_column("protocols", sa.Column("created_by", sa.Uuid(), nullable=True))
    op.execute(BACKFILL_PROTOCOL_CREATORS)


def downgrade() -> None:
    op.drop_column("protocols", "created_by")
    op.drop_column("datasets", "created_by")
```

Add to `backend/tests/integration/test_migrations.py`:

```python
async def test_011_backfills_each_protocols_creator_from_its_training_run(migrated_session):
    spec = importlib.util.spec_from_file_location(
        "migration_011", Path("alembic/versions/011_created_by.py")
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    workspace, protocol, trainer = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO protocols (id, workspace_id, name, dataset_id, engine_id, artifact_uri,"
            " readouts, conditions, status, protocol_version, version, created_at, updated_at)"
            " VALUES (:id, :ws, 'p', :ds, 'rf', 'x', '[]', '{}', 'draft', 1, 1, now(), now())"
        ),
        {"id": protocol, "ws": workspace, "ds": uuid.uuid4()},
    )
    await migrated_session.execute(
        text(
            "INSERT INTO runs (id, workspace_id, kind, requested_by, cache_key, params,"
            " protocol_id, status, progress, attempts, version, created_at, updated_at)"
            " VALUES (:id, :ws, 'training', :by, 'k', '{}', :p, 'ready', 1, 0, 1, now(), now())"
        ),
        {"id": uuid.uuid4(), "ws": workspace, "by": trainer, "p": protocol},
    )

    await migrated_session.execute(text(migration.BACKFILL_PROTOCOL_CREATORS))

    created_by = await migrated_session.scalar(
        text("SELECT created_by FROM protocols WHERE id = :id"), {"id": protocol}
    )
    assert created_by == trainer
```

with these imports added at the top of that file: `import importlib.util`, `import uuid`, `from pathlib import Path`.

- [ ] **Step 9: Record the creator at creation**

In `application/data/create_dataset.py`, in the `Dataset(...)` call, after `validation_report=report,`:

```python
            created_by=auth.user_id,
```

In `application/execution/train_protocol.py`, in the `InSilicoProtocol(...)` call, after `conditions=conditions,`:

```python
                # The person who asked for the training, not the runner that ran it.
                created_by=run.requested_by,
```

- [ ] **Step 10: Expose `can_delete` on both responses**

In `interface/routes/datasets.py`, import `from daikonstudio.application.auth import AuthContext, may_delete`, add the field after `created_at: datetime` in `DatasetResponse`:

```python
    # Whether this viewer may delete it, by role and creator. Dependents (protocols
    # trained on it, runs in progress) are checked only when DELETE is requested.
    can_delete: bool
```

and change `from_domain`:

```python
    @classmethod
    def from_domain(cls, dataset: Dataset, *, auth: AuthContext | None) -> DatasetResponse:
```

with `can_delete=may_delete(auth, dataset.created_by),` as the last argument of `cls(...)`. Pass `auth=auth` at all three callers (`create_dataset`, `list_datasets`, `get_dataset` routes).

In `interface/routes/protocols.py`, the same import, the field after `created_at: datetime` in `ProtocolResponse`:

```python
    # Whether this viewer may delete it: a draft, and an admin or its creator.
    can_delete: bool
```

`from_domain(cls, protocol: InSilicoProtocol, *, auth: AuthContext | None)` with
`can_delete=not protocol.is_locked and may_delete(auth, protocol.created_by),`, and `auth=auth` at both callers (`list_protocols`, `get_protocol`).

- [ ] **Step 11: Run the task's tests**

Run: `cd backend && uv run pytest tests/unit/test_may_delete.py tests/api/test_deleting.py tests/integration/test_migrations.py -q`
Expected: all pass, including `test_no_table_exists_only_in_the_orm_or_only_in_the_migrations`.

- [ ] **Step 12: Run the backend gates and commit**

Run the five backend gates from Global Constraints. Expected: all clean.

```bash
git add backend/src backend/alembic/versions/011_created_by.py backend/tests
git commit -m "feat: record who created datasets and protocols, and say who may delete them"
```

---

### Task 2: Delete a draft protocol (backend)

**Files:**
- Modify: `backend/src/daikonstudio/application/ports/blob_store.py`
- Modify: `backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py`
- Modify: `backend/src/daikonstudio/application/ports/protocol_repository.py`
- Modify: `backend/src/daikonstudio/application/ports/run_repository.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/repository.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py`
- Create: `backend/src/daikonstudio/application/catalog/delete_protocol.py`
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Modify: `backend/tests/integration/test_blob_store.py`
- Modify: `backend/tests/api/test_deleting.py`

**Interfaces:**
- Consumes: `require_may_delete`, `InSilicoProtocol.created_by` (Task 1); `_train_protocol`, `other_editor_client` (Task 1).
- Produces: `BlobStore.delete_prefix(prefix: str) -> None` (prefix must end in `/`; a missing folder is a no-op).
- Produces: `ProtocolRepository.delete(workspace_id, protocol_id) -> None`.
- Produces: `RunRepository.delete_many(workspace_id: UUID, run_ids: Sequence[UUID]) -> None`.
- Produces: `DELETE /api/v1/protocols/{protocol_id}` → 204.

- [ ] **Step 1: Write the failing folder-delete tests**

Add to `backend/tests/integration/test_blob_store.py`:

```python
def test_delete_prefix_removes_one_folder_and_nothing_beside_it(tmp_path):
    store = FsspecBlobStore(f"file://{tmp_path}")
    store.put_bytes("ws/protocols/abc/artifact/model.joblib", b"model")
    store.put_bytes("ws/protocols/abc/scorecard-inputs.json", b"inputs")
    store.put_bytes("ws/protocols/abcdef/scorecard-inputs.json", b"keep")

    store.delete_prefix("ws/protocols/abc/")

    assert not store.exists("ws/protocols/abc/artifact/model.joblib")
    assert not store.exists("ws/protocols/abc/scorecard-inputs.json")
    assert store.exists("ws/protocols/abcdef/scorecard-inputs.json")


def test_delete_prefix_of_a_missing_folder_is_a_no_op(tmp_path):
    FsspecBlobStore(f"file://{tmp_path}").delete_prefix("ws/protocols/never-written/")


def test_delete_prefix_refuses_a_key_that_is_not_a_folder(tmp_path):
    with pytest.raises(ValueError, match="ending in '/'"):
        FsspecBlobStore(f"file://{tmp_path}").delete_prefix("ws/protocols/abc")
```

(Import `FsspecBlobStore` and `pytest` at the top of the file if they are not already imported.)

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && uv run pytest tests/integration/test_blob_store.py -q -k delete_prefix`
Expected: 3 failed with `AttributeError: 'FsspecBlobStore' object has no attribute 'delete_prefix'`.

- [ ] **Step 3: Implement `delete_prefix`**

Port, `application/ports/blob_store.py`, after `delete`:

```python
    def delete_prefix(self, prefix: str) -> None:
        """Every blob under a folder key ending in "/". A missing folder is a no-op."""
        ...
```

`infrastructure/storage/fsspec_blob_store.py`, after `delete`:

```python
    def delete_prefix(self, prefix: str) -> None:
        # The trailing "/" is what keeps `ws/protocols/abc/` from also matching
        # `ws/protocols/abcdef/` on a store whose "folders" are key prefixes (S3).
        if not prefix.endswith("/"):
            raise ValueError(f"delete_prefix needs a folder key ending in '/', got {prefix!r}")
        path = self._path(prefix).rstrip("/")
        if self._fs.exists(path):
            self._fs.rm(path, recursive=True)
```

- [ ] **Step 4: Run them to verify they pass**

Run: `cd backend && uv run pytest tests/integration/test_blob_store.py -q -k delete_prefix`
Expected: 3 passed.

- [ ] **Step 5: Write the failing API tests**

Add to `backend/tests/api/test_deleting.py` (add `from sqlalchemy import text` to its imports):

```python
async def test_the_creator_deletes_a_draft_with_its_training_run_and_files(
    client, csv_upload, workspace_id, tmp_path
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _train_protocol(client, dataset_id)
    folder = tmp_path / str(workspace_id) / "protocols" / protocol_id
    assert folder.exists()

    response = await client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 204, response.text
    assert (await client.get(f"/api/v1/protocols/{protocol_id}")).status_code == 404
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404
    assert not folder.exists()
    # A second delete finds nothing: 404, not 500.
    assert (await client.delete(f"/api/v1/protocols/{protocol_id}")).status_code == 404


async def test_a_published_protocol_cannot_be_deleted(client, admin_client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    assert (await client.post(f"/api/v1/protocols/{protocol_id}/publish")).status_code == 204

    response = await admin_client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 409
    assert response.json()["message"] == "Published protocols cannot be deleted."


async def test_only_the_creator_or_an_admin_may_delete_a_draft(
    client, other_editor_client, viewer_client, other_workspace_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    url = f"/api/v1/protocols/{protocol_id}"

    refused = await other_editor_client.delete(url)
    assert refused.status_code == 403
    assert refused.json()["message"] == (
        "Only an admin or the person who created it can delete this."
    )
    assert (await viewer_client.delete(url)).status_code == 403
    assert (await other_workspace_client.delete(url)).status_code == 404
    assert (await admin_client.delete(url)).status_code == 204


async def test_a_draft_whose_training_run_is_still_finishing_is_refused(
    client, csv_upload, session_factory
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _train_protocol(client, dataset_id)
    # The protocol row exists before the run's last phase ("Mapping chemical space") ends.
    async with session_factory() as session:
        await session.execute(
            text("UPDATE runs SET status = 'running' WHERE id = :id"), {"id": run_id}
        )
        await session.commit()

    response = await client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 409
    assert "still finishing" in response.json()["message"]
```

- [ ] **Step 6: Run them to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_deleting.py -q -k "draft or published"`
Expected: the four new tests fail with `405 == 204`-style assertion errors (no DELETE route yet).

- [ ] **Step 7: Add the repository deletes**

`application/ports/protocol_repository.py`, after `update`:

```python
    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None: ...
```

and append to its module docstring: "`delete` removes a draft; `DeleteProtocol` is the only caller and refuses a published one."

`infrastructure/persistence/sqlalchemy/catalog/repository.py` (import `delete as sa_delete` from `sqlalchemy`), after `update`:

```python
    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_delete(InSilicoProtocolModel).where(
                    InSilicoProtocolModel.id == protocol_id,
                    InSilicoProtocolModel.workspace_id == workspace_id,
                )
            )
            await session.commit()
```

`application/ports/run_repository.py` (import `Sequence` from `collections.abc`), after `update`:

```python
    async def delete_many(self, workspace_id: UUID, run_ids: Sequence[UUID]) -> None:
        """Only for the runs of something being deleted: a draft Protocol's training
        run, or a Dataset's failed and cancelled training runs."""
        ...
```

`infrastructure/persistence/sqlalchemy/execution/repository.py` (import `delete as sa_delete` from `sqlalchemy`, `Sequence` from `collections.abc`), after `update`:

```python
    async def delete_many(self, workspace_id: uuid.UUID, run_ids: Sequence[uuid.UUID]) -> None:
        if not run_ids:
            return
        async with self._sessions() as session:
            await session.execute(
                sa_delete(RunModel).where(
                    RunModel.workspace_id == workspace_id, RunModel.id.in_(run_ids)
                )
            )
            await session.commit()
```

Also update the comment in `application/execution/claim_run.py:67` that says nothing deletes a Run: replace "nothing in this codebase deletes a Run." with "only DeleteProtocol and DeleteDataset delete Runs, and never a pending or running one."

- [ ] **Step 8: Write the use case**

`backend/src/daikonstudio/application/catalog/delete_protocol.py`:

```python
"""Delete a draft Protocol, with the training run that produced it and its files.

Only a draft. A published Protocol is citable, and predictions and collections
may depend on it. A draft has neither (a prediction needs a published Protocol,
and `new_version` needs a published parent), so its training run and its folder
of files are everything that depends on it.

Rows go first, then files: a failed file delete leaves an orphan folder, which is
harmless, where the reverse order could leave a row pointing at files that are
gone. Runs go before the Protocol, so a failure between the two leaves a draft
that can simply be deleted again.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_may_delete,
)
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunStatus
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError

logger = logging.getLogger(__name__)

_ACTIVE = {RunStatus.PENDING, RunStatus.RUNNING}


def protocol_folder(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Every file a Protocol owns lives under this folder: the artifact, the scorecard
    inputs and the chemical-space map. Never delete `artifact_uri` on its own: a
    versioned Protocol's can point into another Protocol's folder."""
    return f"{workspace_id}/protocols/{protocol_id}/"


@dataclass(frozen=True, kw_only=True)
class DeleteProtocolCommand:
    protocol_id: uuid.UUID


class DeleteProtocol:
    def __init__(
        self, protocols: ProtocolRepository, runs: RunRepository, store: BlobStore
    ) -> None:
        self._protocols = protocols
        self._runs = runs
        self._store = store

    async def __call__(
        self, command: DeleteProtocolCommand, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await self._protocols.get(auth.workspace_id, command.protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.protocol_id)))
        require_may_delete(auth, protocol.created_by)
        if protocol.is_locked:
            return Failure(ConflictError("Published protocols cannot be deleted."))

        # A draft has one training run; the limit is a ceiling, not a page size.
        runs = await self._runs.list(auth.workspace_id, protocol_id=protocol.id, limit=100)
        if any(run.status in _ACTIVE for run in runs):
            return Failure(
                ConflictError(
                    "This protocol's training run is still finishing. "
                    "Try again when it completes."
                )
            )

        await self._runs.delete_many(auth.workspace_id, [run.id for run in runs])
        await self._protocols.delete(auth.workspace_id, protocol.id)
        folder = protocol_folder(auth.workspace_id, protocol.id)
        try:
            self._store.delete_prefix(folder)
        except Exception:
            logger.exception("Deleting %s failed; the folder is orphaned", folder)
        logger.info("Protocol %s deleted by %s", protocol.id, auth.user_id)
        return Success(None)
```

- [ ] **Step 9: Add the route and the wiring**

`interface/routes/protocols.py`: import `DeleteProtocol, DeleteProtocolCommand` from `daikonstudio.application.catalog.delete_protocol`, add next to `PublishProtocolDep`:

```python
DeleteProtocolDep = Annotated[DeleteProtocol, Depends(use_case(DeleteProtocol))]
```

and after `publish_protocol`:

```python
@router.delete("/{protocol_id}", status_code=204)
async def delete_protocol(
    protocol_id: uuid.UUID, auth: AuthDep, service: DeleteProtocolDep
) -> Response:
    """A draft only, by an admin or its creator. Also deletes the training run that
    produced it and every file in its folder. See `application/catalog/delete_protocol.py`."""
    result_to_response(await service(DeleteProtocolCommand(protocol_id=protocol_id), auth=auth))
    return Response(status_code=204)
```

`infrastructure/di/container.py`: import `DeleteProtocol` and, next to the `PublishProtocol` definition:

```python
    container.define(
        DeleteProtocol, lambda c: DeleteProtocol(_protocols(c), _runs(c), c[BlobStore])
    )
```

- [ ] **Step 10: Run the task's tests**

Run: `cd backend && uv run pytest tests/api/test_deleting.py tests/integration/test_blob_store.py -q`
Expected: all pass.

- [ ] **Step 11: Run the backend gates and commit**

Run the five backend gates. Expected: all clean.

```bash
git add backend/src backend/tests
git commit -m "feat(protocols): delete a draft protocol with its training run and files"
```

---

### Task 3: Delete a dataset (backend)

**Files:**
- Modify: `backend/src/daikonstudio/application/ports/dataset_repository.py`
- Modify: `backend/src/daikonstudio/application/ports/protocol_repository.py` (`list`)
- Modify: `backend/src/daikonstudio/application/ports/run_repository.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/repository.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/repository.py` (`list`)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py`
- Modify: `backend/src/daikonstudio/application/catalog/list_protocols.py` (`ListProtocolsQuery`, `ListProtocols`)
- Create: `backend/src/daikonstudio/application/data/delete_dataset.py`
- Modify: `backend/src/daikonstudio/application/data/get_dataset_profile.py` (`_compute`)
- Modify: `backend/src/daikonstudio/interface/routes/datasets.py` (module docstring, new route)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py` (`list_protocols` filter)
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Modify: `backend/tests/api/test_deleting.py`

**Interfaces:**
- Consumes: `require_may_delete`, `Dataset.created_by` (Task 1); `BlobStore.delete_prefix`, `RunRepository.delete_many` (Task 2).
- Produces: `DatasetRepository.delete(workspace_id, dataset_id) -> None`.
- Produces: `ProtocolRepository.list(..., dataset_id: uuid.UUID | None = None)`; `GET /api/v1/protocols?dataset_id=`.
- Produces: `RunRepository.list_training_for_dataset(workspace_id: UUID, dataset_id: UUID) -> builtins.list[Run]`.
- Produces: `DELETE /api/v1/datasets/{dataset_id}` → 204.

- [ ] **Step 1: Write the failing API tests**

Add to `backend/tests/api/test_deleting.py` (imports: `asyncio`, `threading`, `uuid`; `from daikonstudio.application.data import get_dataset_profile`; `from daikonstudio.domain.execution.run import Run, RunKind, RunStatus`; `from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import SqlAlchemyRunRepository`):

```python
async def _add_training_run(session_factory, workspace_id, dataset_id: str, status) -> str:
    """A training run on the dataset, in a state the inline test enqueuer never leaves one."""
    run = Run(
        kind=RunKind.TRAINING,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="test",
        params={"dataset_id": dataset_id},
        status=status,
    )
    await SqlAlchemyRunRepository(session_factory).add(run)
    return str(run.id)


async def test_the_creator_deletes_an_unused_dataset_and_can_upload_it_again(
    client, csv_upload, workspace_id, tmp_path
):
    dataset_id = await _create_dataset(client, csv_upload)
    folder = tmp_path / str(workspace_id) / "datasets" / dataset_id
    assert folder.exists()

    response = await client.delete(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 204, response.text
    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).status_code == 404
    assert not folder.exists()
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 404
    # The content-hash unique index no longer holds the old row.
    assert await _create_dataset(client, csv_upload) != dataset_id


async def test_a_dataset_with_a_protocol_is_refused_until_the_draft_is_deleted(
    client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)

    refused = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert refused.status_code == 409
    assert refused.json()["message"] == (
        "Protocols trained on this dataset must be deleted first. "
        "A dataset used by a published protocol cannot be deleted."
    )
    listing = await client.get("/api/v1/protocols", params={"dataset_id": dataset_id})
    assert [item["id"] for item in listing.json()["items"]] == [protocol_id]

    assert (await client.delete(f"/api/v1/protocols/{protocol_id}")).status_code == 204
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204


async def test_protocols_can_be_listed_by_the_dataset_they_were_trained_on(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)

    mine = await client.get("/api/v1/protocols", params={"dataset_id": dataset_id})
    other = await client.get("/api/v1/protocols", params={"dataset_id": str(uuid.uuid4())})

    assert [item["id"] for item in mine.json()["items"]] == [protocol_id]
    # Without the filter this would list every protocol in the workspace.
    assert other.json()["items"] == []


async def test_a_dataset_with_a_training_run_in_progress_is_refused(
    client, csv_upload, session_factory, workspace_id
):
    dataset_id = await _create_dataset(client, csv_upload)
    await _add_training_run(session_factory, workspace_id, dataset_id, RunStatus.PENDING)

    response = await client.delete(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 409
    assert response.json()["message"] == "A training run on this dataset is still in progress."


async def test_failed_training_runs_go_with_their_dataset(
    client, csv_upload, session_factory, workspace_id
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id = await _add_training_run(session_factory, workspace_id, dataset_id, RunStatus.FAILED)

    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404


async def test_only_the_creator_or_an_admin_may_delete_a_dataset(
    client, other_editor_client, viewer_client, other_workspace_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    url = f"/api/v1/datasets/{dataset_id}"

    assert (await other_editor_client.delete(url)).status_code == 403
    assert (await viewer_client.delete(url)).status_code == 403
    assert (await other_workspace_client.delete(url)).status_code == 404
    assert (await admin_client.delete(url)).status_code == 204


async def test_a_dataset_deleted_while_its_profile_computes_stays_deleted(
    client, csv_upload, monkeypatch, workspace_id, tmp_path
):
    release = threading.Event()
    real = get_dataset_profile.build_profile

    def slow_build(**kwargs):
        release.wait(timeout=10)
        return real(**kwargs)

    monkeypatch.setattr(get_dataset_profile, "build_profile", slow_build)
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await client.get(f"/api/v1/datasets/{dataset_id}/profile")).status_code == 202

    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204
    release.set()
    for _ in range(200):
        if not get_dataset_profile._RUNNING:
            break
        await asyncio.sleep(0.05)

    assert not (tmp_path / str(workspace_id) / "datasets" / dataset_id).exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_deleting.py -q -k "dataset"`
Expected: the seven new tests fail: 405 from the missing DELETE route, and the filter test lists a protocol for an unknown dataset.

- [ ] **Step 3: Add the repository methods**

`application/ports/dataset_repository.py`: change the docstring's last sentence to "There is no `update`: a Dataset is immutable once created. `delete` removes one that nothing depends on; `DeleteDataset` is the only caller and checks that." and add:

```python
    async def delete(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> None: ...
```

`infrastructure/persistence/sqlalchemy/data/repository.py` (import `delete as sa_delete`):

```python
    async def delete(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_delete(DatasetModel).where(
                    DatasetModel.id == dataset_id, DatasetModel.workspace_id == workspace_id
                )
            )
            await session.commit()
```

`ProtocolRepository.list` (port and implementation) gains a keyword `dataset_id: uuid.UUID | None = None` after `limit`; the implementation adds, before the cursor clause:

```python
        if dataset_id is not None:
            statement = statement.where(InSilicoProtocolModel.dataset_id == dataset_id)
```

`application/ports/run_repository.py`, after `delete_many`:

```python
    async def list_training_for_dataset(
        self, workspace_id: UUID, dataset_id: UUID
    ) -> builtins.list[Run]:
        """Every training run on a Dataset. Unpaginated: they are submitted by hand."""
        ...
```

`infrastructure/persistence/sqlalchemy/execution/repository.py`:

```python
    async def list_training_for_dataset(
        self, workspace_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> builtins.list[Run]:
        # ponytail: a JSONB scan within the workspace, no index. Fine while a
        # workspace has thousands of runs; add an expression index on
        # (workspace_id, params->>'dataset_id') if deletes get slow.
        statement = select(RunModel).where(
            RunModel.workspace_id == workspace_id,
            RunModel.kind == RunKind.TRAINING.value,
            RunModel.params["dataset_id"].astext == str(dataset_id),
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [_to_domain(model) for model in result.scalars()]
```

`application/catalog/list_protocols.py`: `ListProtocolsQuery` gains `dataset_id: uuid.UUID | None = None`; `ListProtocols.__call__` passes `dataset_id=query.dataset_id` to `self._repository.list`.

`interface/routes/protocols.py` `list_protocols`: add the parameter `dataset_id: uuid.UUID | None = None` and pass `dataset_id=dataset_id` into `ListProtocolsQuery`.

- [ ] **Step 4: Write the use case**

`backend/src/daikonstudio/application/data/delete_dataset.py`:

```python
"""Delete a Dataset that nothing depends on.

A Dataset is immutable and cited by id, which is why one that anything still
depends on cannot be deleted: any Protocol trained on it (a published one reads
its training compounds on every prediction run, to measure applicability
domain), and any training run on it still pending or running. Training runs that
failed or were cancelled have no Protocol, and are deleted with it.

Rows go first, then files, for the reason given in `delete_protocol.py`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_may_delete,
)
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunStatus
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError

logger = logging.getLogger(__name__)

_ACTIVE = {RunStatus.PENDING, RunStatus.RUNNING}

PROTOCOLS_FIRST = (
    "Protocols trained on this dataset must be deleted first. "
    "A dataset used by a published protocol cannot be deleted."
)


def dataset_folder(workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    """The snapshot and the profile both live under this folder (`snapshot.py`,
    `get_dataset_profile.py`)."""
    return f"{workspace_id}/datasets/{dataset_id}/"


@dataclass(frozen=True, kw_only=True)
class DeleteDatasetCommand:
    dataset_id: uuid.UUID


class DeleteDataset:
    def __init__(
        self,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._store = store

    async def __call__(
        self, command: DeleteDatasetCommand, auth: AuthContext | None = None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        require_may_delete(auth, dataset.created_by)
        if await self._protocols.list(auth.workspace_id, dataset_id=dataset.id, limit=1):
            return Failure(ConflictError(PROTOCOLS_FIRST))
        runs = await self._runs.list_training_for_dataset(auth.workspace_id, dataset.id)
        if any(run.status in _ACTIVE for run in runs):
            return Failure(ConflictError("A training run on this dataset is still in progress."))

        # ponytail: a training run submitted between the check above and the
        # delete below fails when it reads the missing snapshot. Rare (it needs
        # two people acting on one dataset in the same second), and loud.
        await self._runs.delete_many(auth.workspace_id, [run.id for run in runs])
        await self._datasets.delete(auth.workspace_id, dataset.id)
        folder = dataset_folder(auth.workspace_id, dataset.id)
        try:
            self._store.delete_prefix(folder)
        except Exception:
            logger.exception("Deleting %s failed; the folder is orphaned", folder)
        logger.info("Dataset %s deleted by %s", dataset.id, auth.user_id)
        return Success(None)
```

- [ ] **Step 5: Keep a deleted dataset's profile from recreating its folder**

In `application/data/get_dataset_profile.py`, `_compute`, immediately before `self._store.put_bytes(profile_key(...), ...)`:

```python
            if not self._store.exists(snapshot_key(dataset.workspace_id, dataset.id)):
                # Deleted while this ran: saving would recreate its folder.
                return
```

- [ ] **Step 6: Add the route and the wiring**

`interface/routes/datasets.py`: replace "There is no PATCH and no DELETE. A Dataset is immutable and cited by id." in the module docstring with "There is no PATCH: a Dataset is immutable and cited by id. DELETE removes one only when nothing depends on it; see `application/data/delete_dataset.py`." Import `DeleteDataset, DeleteDatasetCommand`, add

```python
DeleteDatasetDep = Annotated[DeleteDataset, Depends(use_case(DeleteDataset))]
```

and after `get_dataset`:

```python
@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: uuid.UUID, auth: AuthDep, service: DeleteDatasetDep) -> Response:
    """By an admin or its creator, and only when no protocol was trained on it and no
    training run on it is in progress. See `application/data/delete_dataset.py`."""
    result_to_response(await service(DeleteDatasetCommand(dataset_id=dataset_id), auth=auth))
    return Response(status_code=204)
```

(import `Response` from `fastapi` if the module does not already.)

`infrastructure/di/container.py`: import `DeleteDataset` and define, next to `GetDatasetProfile`:

```python
    container.define(
        DeleteDataset,
        lambda c: DeleteDataset(_datasets(c), _protocols(c), _runs(c), c[BlobStore]),
    )
```

- [ ] **Step 7: Run the task's tests**

Run: `cd backend && uv run pytest tests/api/test_deleting.py tests/api/test_dataset_profile.py -q`
Expected: all pass.

- [ ] **Step 8: Run the backend gates and commit**

Run the five backend gates. Expected: all clean.

```bash
git add backend/src backend/tests
git commit -m "feat(datasets): delete a dataset that nothing depends on"
```

---

### Task 4: Delete a draft protocol from its page

**Files:**
- Regenerate: `frontend/openapi.json`, `frontend/src/shared/lib/api/**` via `make generate-api`
- Modify: `frontend/src/features/protocols/hooks/use-protocols.ts`
- Create: `frontend/src/features/protocols/components/delete-protocol-button.tsx`
- Create: `frontend/src/features/protocols/components/delete-protocol-button.test.tsx`
- Modify: `frontend/src/features/protocols/components/protocol-detail.tsx`

**Interfaces:**
- Consumes: `DELETE /api/v1/protocols/{id}`, `ProtocolResponse.can_delete` (Tasks 1–2).
- Produces: `useDeleteProtocol()` mutation taking a protocol id; `DeleteProtocolButton({ protocol }: { protocol: ProtocolResponse })`.

- [ ] **Step 1: Regenerate the client**

Run: `make generate-api` from the repo root.
Expected: `ProtocolResponse` and `DatasetResponse` in `frontend/src/shared/lib/api/model` now carry `can_delete: boolean`.

- [ ] **Step 2: Write the failing component test**

`frontend/src/features/protocols/components/delete-protocol-button.test.tsx`:

```tsx
import type { ProtocolResponse } from "@/shared/lib/api/model";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeleteProtocolButton } from "./delete-protocol-button";

const hoisted = vi.hoisted(() => ({
  push: vi.fn(),
  mutate: vi.fn(),
  error: null as Error | null,
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: hoisted.push }) }));
vi.mock("../hooks/use-protocols", () => ({
  useDeleteProtocol: () => ({
    mutate: hoisted.mutate,
    reset: vi.fn(),
    isPending: false,
    error: hoisted.error,
  }),
}));

const PROTOCOL = { id: "p-1", name: "solubility rf", can_delete: true } as ProtocolResponse;

describe("DeleteProtocolButton", () => {
  beforeEach(() => {
    hoisted.push.mockReset();
    hoisted.mutate.mockReset();
    hoisted.error = null;
  });

  it("deletes on confirmation, then returns to the protocols list", () => {
    hoisted.mutate.mockImplementation((_id: string, options: { onSuccess: () => void }) =>
      options.onSuccess(),
    );
    render(<DeleteProtocolButton protocol={PROTOCOL} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(screen.getByText(/the training run that produced it/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));

    expect(hoisted.mutate).toHaveBeenCalledWith("p-1", expect.anything());
    expect(hoisted.push).toHaveBeenCalledWith("/protocols");
  });

  it("shows the server's refusal inside the dialog", () => {
    hoisted.error = new Error("Published protocols cannot be deleted.");
    render(<DeleteProtocolButton protocol={PROTOCOL} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Published protocols cannot be deleted.");
  });
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd frontend && pnpm test src/features/protocols/components/delete-protocol-button`
Expected: FAIL with `Failed to resolve import "./delete-protocol-button"`.

- [ ] **Step 4: Add the hook**

In `frontend/src/features/protocols/hooks/use-protocols.ts`, after `usePublishProtocol`:

```ts
export function useDeleteProtocol() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/protocols/${id}`, method: "DELETE" }),
    // The dialog shows the error; no second toast.
    meta: { silent: true },
    onSuccess: () => {
      showSuccess("Protocol deleted.");
      // Lists, its training run, its sweep and its dataset's dialog all change.
      // Mark everything stale without refetching: the page being left would
      // otherwise refetch the protocol it just deleted and flash a 404.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}
```

- [ ] **Step 5: Write the component**

`frontend/src/features/protocols/components/delete-protocol-button.tsx`:

```tsx
"use client";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/shared/components/ui/alert-dialog";
import { Button, buttonVariants } from "@/shared/components/ui/button";
import type { ProtocolResponse } from "@/shared/lib/api/model";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useDeleteProtocol } from "../hooks/use-protocols";

export function DeleteProtocolButton({ protocol }: { protocol: ProtocolResponse }) {
  const router = useRouter();
  const remove = useDeleteProtocol();
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button
        variant="outline"
        onClick={() => {
          remove.reset();
          setOpen(true);
        }}
      >
        Delete
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete draft protocol “{protocol.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes the trained model, its scorecard and chemical-space map,
              and the training run that produced it. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {remove.error && (
            <p role="alert" className="text-sm text-destructive">
              {remove.error.message}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: "destructive" })}
              disabled={remove.isPending}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                remove.mutate(protocol.id, {
                  onSuccess: () => {
                    setOpen(false);
                    router.push("/protocols");
                  },
                });
              }}
            >
              {remove.isPending ? "Deleting…" : "Delete permanently"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
```

In `protocol-detail.tsx`, import it and render it first inside `<div className="flex gap-2">`:

```tsx
          {protocol.can_delete && <DeleteProtocolButton protocol={protocol} />}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd frontend && pnpm test src/features/protocols/components/delete-protocol-button`
Expected: 2 passed.

- [ ] **Step 7: Run the frontend gates and commit**

Run the three frontend gates. Expected: all clean.

```bash
git add frontend
git commit -m "feat(protocols): delete a draft protocol from its page"
```

---

### Task 5: Delete a dataset from its page

**Files:**
- Modify: `frontend/src/features/datasets/hooks/use-datasets.ts`
- Create: `frontend/src/features/datasets/components/delete-dataset-button.tsx`
- Create: `frontend/src/features/datasets/components/delete-dataset-button.test.tsx`
- Modify: `frontend/src/features/datasets/components/dataset-detail.tsx`

**Interfaces:**
- Consumes: `DELETE /api/v1/datasets/{id}`, `GET /api/v1/protocols?dataset_id=`, `DatasetResponse.can_delete` (Tasks 1, 3, 4's regenerated client).
- Produces: `useDeleteDataset()`, `useDatasetProtocols(datasetId: string, enabled: boolean)`, `DeleteDatasetButton({ dataset }: { dataset: DatasetResponse })`.

- [ ] **Step 1: Write the failing component test**

`frontend/src/features/datasets/components/delete-dataset-button.test.tsx`:

```tsx
import type { DatasetResponse } from "@/shared/lib/api/model";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeleteDatasetButton } from "./delete-dataset-button";

const hoisted = vi.hoisted(() => ({
  push: vi.fn(),
  mutate: vi.fn(),
  protocols: [] as { id: string; name: string; status: string; protocol_version: number }[],
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: hoisted.push }) }));
vi.mock("../hooks/use-datasets", () => ({
  useDeleteDataset: () => ({ mutate: hoisted.mutate, reset: vi.fn(), isPending: false, error: null }),
  useDatasetProtocols: () => ({ isLoading: false, data: { items: hoisted.protocols } }),
}));

const DATASET = { id: "d-1", name: "ben-inhibition", can_delete: true } as DatasetResponse;

describe("DeleteDatasetButton", () => {
  beforeEach(() => {
    hoisted.push.mockReset();
    hoisted.mutate.mockReset();
    hoisted.protocols = [];
  });

  it("deletes an unused dataset, then returns to the datasets list", () => {
    hoisted.mutate.mockImplementation((_id: string, options: { onSuccess: () => void }) =>
      options.onSuccess(),
    );
    render(<DeleteDatasetButton dataset={DATASET} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));

    expect(hoisted.mutate).toHaveBeenCalledWith("d-1", expect.anything());
    expect(hoisted.push).toHaveBeenCalledWith("/datasets");
  });

  it("lists the protocols trained on it and will not delete until they are gone", () => {
    hoisted.protocols = [
      { id: "p-1", name: "solubility rf", status: "draft", protocol_version: 1 },
      { id: "p-2", name: "solubility gp", status: "published", protocol_version: 2 },
    ];
    render(<DeleteDatasetButton dataset={DATASET} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(screen.getByText(/must be deleted first/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "solubility rf" })).toHaveAttribute(
      "href",
      "/protocols/p-1",
    );
    expect(screen.getByRole("button", { name: "Delete permanently" })).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && pnpm test src/features/datasets/components/delete-dataset-button`
Expected: FAIL with `Failed to resolve import "./delete-dataset-button"`.

- [ ] **Step 3: Add the hooks**

In `frontend/src/features/datasets/hooks/use-datasets.ts` (import `PaginatedResponseProtocolResponse` from `@/shared/lib/api/model`):

```ts
/**
 * The protocols trained on a dataset: what stands between it and deletion.
 * Keyed under "protocols" (the protocols feature's root key, not imported to
 * keep the two features' barrels from importing each other) so anything that
 * invalidates protocols refreshes this too.
 */
export function useDatasetProtocols(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["protocols", "by-dataset", datasetId],
    queryFn: () =>
      customInstance<PaginatedResponseProtocolResponse>({
        url: `${API_V1}/protocols`,
        method: "GET",
        params: { dataset_id: datasetId, limit: 200 },
      }),
    enabled,
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/datasets/${id}`, method: "DELETE" }),
    // The dialog shows the error; no second toast.
    meta: { silent: true },
    onSuccess: () => {
      showSuccess("Dataset deleted.");
      // Mark everything stale without refetching: the page being left would
      // otherwise refetch the dataset it just deleted and flash a 404.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}
```

- [ ] **Step 4: Write the component**

`frontend/src/features/datasets/components/delete-dataset-button.tsx`:

```tsx
"use client";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/shared/components/ui/alert-dialog";
import { Badge } from "@/shared/components/ui/badge";
import { Button, buttonVariants } from "@/shared/components/ui/button";
import type { DatasetResponse } from "@/shared/lib/api/model";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useDatasetProtocols, useDeleteDataset } from "../hooks/use-datasets";

export function DeleteDatasetButton({ dataset }: { dataset: DatasetResponse }) {
  const router = useRouter();
  const remove = useDeleteDataset();
  const [open, setOpen] = useState(false);
  // Fetched when the dialog opens, so it reflects any protocol deleted since.
  const protocols = useDatasetProtocols(dataset.id, open);
  const blockers = protocols.data?.items ?? [];

  return (
    <>
      <Button
        variant="outline"
        onClick={() => {
          remove.reset();
          setOpen(true);
        }}
      >
        Delete
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{dataset.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes the dataset's frozen snapshot and profile. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {protocols.isLoading && (
            <p className="text-sm text-muted-foreground">
              Checking for protocols trained on this dataset…
            </p>
          )}
          {blockers.length > 0 && (
            <div className="space-y-2 text-sm">
              <p>
                Protocols trained on this dataset must be deleted first. A dataset used by a
                published protocol cannot be deleted.
              </p>
              <ul className="space-y-1">
                {blockers.map((protocol) => (
                  <li key={protocol.id} className="flex items-center gap-2">
                    <Link
                      href={`/protocols/${protocol.id}`}
                      className="underline underline-offset-2"
                    >
                      {protocol.name}
                    </Link>
                    <span className="text-muted-foreground">v{protocol.protocol_version}</span>
                    <Badge
                      variant={protocol.status === "draft" ? "outline" : "default"}
                      className="font-normal"
                    >
                      {protocol.status === "draft" ? "Draft" : "Published"}
                    </Badge>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {remove.error && (
            <p role="alert" className="text-sm text-destructive">
              {remove.error.message}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: "destructive" })}
              disabled={remove.isPending || protocols.isLoading || blockers.length > 0}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                remove.mutate(dataset.id, {
                  onSuccess: () => {
                    setOpen(false);
                    router.push("/datasets");
                  },
                });
              }}
            >
              {remove.isPending ? "Deleting…" : "Delete permanently"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
```

In `dataset-detail.tsx`, import it and wrap the header's "Train a protocol" button:

```tsx
        <div className="flex gap-2">
          {dataset.can_delete && <DeleteDatasetButton dataset={dataset} />}
          <Button asChild>
            <Link href={`/protocols/new?dataset=${dataset.id}`}>Train a protocol</Link>
          </Button>
        </div>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && pnpm test src/features/datasets/components/delete-dataset-button`
Expected: 2 passed.

- [ ] **Step 6: Run the frontend gates and commit**

Run the three frontend gates. Expected: all clean.

```bash
git add frontend
git commit -m "feat(datasets): delete a dataset from its page"
```

---

## After the tasks

- Final whole-branch review on the most capable model, with this plan's Review Focus.
- Live check in dev, each step asked of the user first: upload a small throwaway dataset, train a draft protocol on it, try deleting the dataset (expect the dialog to list the draft), delete the draft, then delete the dataset.
- The dev API reloads on backend changes; apply migration 011 to the dev database with `make migrate` before the live check.
