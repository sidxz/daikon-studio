# daikon-studio Phase 1 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the daikon-studio backend so a scientist can upload a CSV, train an in-silico Protocol with an honest Scorecard, publish it, run it against new compounds, triage the results, and export a Collection — all over HTTP, with no GPU.

**Architecture:** Clean Architecture in three bounded contexts (`catalog`, `data`, `execution`), cloned from prot-cellar's layout and enforced by import-linter. Engines are in-tree implementations of a structural `Engine` Protocol modelled on prot-cellar's `IngestionPlugin`. Long work runs on arq over Valkey with DB-column progress. Datasets and results are immutable Parquet snapshots on an fsspec blob store, content-addressed so identical work is never repeated.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2.0 async + asyncpg, Alembic, Pydantic v2, Lagom, `returns`, arq, structlog, sentinel-auth-sdk, PostgreSQL 16, Valkey, polars, rdkit, scikit-learn, xgboost, fsspec.

**Source spec:** `docs/superpowers/specs/2026-07-25-daikon-studio-phase-1-design.md`

## Global Constraints

- Python `>=3.13`. Package name `daikonstudio`. Dependency manager `uv`.
- Ports, verified free on the dev machine 2026-07-25: backend **8002**, Postgres **5435**, Valkey **6381**. (5434 was the original choice but is held by `daikon-gen3-postgres-1`; 3002 is also occupied, so the frontend plan must pick 3003 or later. Verify with `lsof -nP -iTCP:<port> -sTCP:LISTEN` before assuming.)
- Import-linter contracts must pass in CI: layer order `interface > infrastructure > application > domain`; domain purity (domain imports none of fastapi, sqlalchemy, asyncpg, redis, lagom, arq, httpx, structlog, polars, rdkit, sklearn, xgboost, fsspec); bounded-context independence at the DOMAIN layer (`domain.catalog`, `domain.data`, `domain.execution` may not import each other — only `domain.shared`). Application-layer orchestration across contexts is expected and allowed; that is where Task 14 lives.
- Use cases return `Result[T, DomainError]` from `returns`. Never raise for expected failures. Guards are the first lines of every use case.
- `workspace_id` comes from `auth.workspace_id`, never from a request body or URL. Every table carries it.
- Every aggregate uses optimistic concurrency: `UPDATE ... WHERE id=? AND version=?`, 0 rows → `ConcurrencyConflictError`.
- Sentinel actions, declared at startup, best-effort (never fatal): `studio:read`, `studio:write`, `studio:train`, `studio:publish`, `studio:admin_config`.
- Ruff for lint and format, mypy strict. `make fmt` before every commit.
- All randomness takes an explicit seed. Splits, model fits, and shuffles must be reproducible from `(content_hash, seed)`.
- Deliberate simplifications get a `ponytail:` comment naming the ceiling and the upgrade path.

---

### Task 1: Repo skeleton and a health endpoint

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/daikonstudio/interface/app.py`
- Create: `backend/src/daikonstudio/settings.py`
- Create: `docker-compose.yml`
- Create: `Makefile`
- Test: `backend/tests/api/test_health.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `create_app() -> FastAPI`; `Settings` (pydantic-settings, env prefix `STUDIO_`) with `database_url: str`, `redis_url: str`, `blob_base_url: str`, `cors_origins: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_health.py
from fastapi.testclient import TestClient

from daikonstudio.interface.app import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_service_name():
    client = TestClient(create_app())
    assert response_json(client)["service"] == "daikon-studio"


def response_json(client):
    return client.get("/version").json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daikonstudio'`

- [ ] **Step 3: Write `pyproject.toml`**

Copy `/Users/sidx/workspace/prot-cellar/backend/pyproject.toml` and change: `name = "daikonstudio"`, `description = "In-silico protocol studio (ML sibling of cellar)"`, `packages = ["src/daikonstudio"]`, and every `protcellar.` prefix in the import-linter contracts to `daikonstudio.`. Replace the domain dependencies `biopython`/`obonet` with:

```toml
    "polars>=1.17",
    "pyarrow>=18.0",
    "rdkit>=2024.09",
    "scikit-learn>=1.6",
    "xgboost>=2.1",
    "fsspec>=2024.10",
    "joblib>=1.4",
```

Add the third import-linter contract:

```toml
[[tool.importlinter.contracts]]
name = "Bounded context independence"
type = "independence"
modules = [
    "daikonstudio.domain.catalog",
    "daikonstudio.domain.data",
    "daikonstudio.domain.execution",
]
```

- [ ] **Step 4: Write settings and the app**

```python
# backend/src/daikonstudio/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://studio:studio@localhost:5435/studio"
    redis_url: str = "redis://localhost:6381"
    blob_base_url: str = "file:///data/blobs"
    cors_origins: list[str] = ["http://localhost:3002"]
    service_name: str = "daikon-studio"
```

```python
# backend/src/daikonstudio/interface/app.py
from fastapi import FastAPI

from daikonstudio.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="daikon-studio", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"service": settings.service_name, "version": app.version}

    return app
```

- [ ] **Step 5: Write `docker-compose.yml` (infra only)**

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment: {POSTGRES_USER: studio, POSTGRES_PASSWORD: studio, POSTGRES_DB: studio}
    ports: ["127.0.0.1:5435:5432"]
    volumes: ["studio_pgdata:/var/lib/postgresql/data"]
  valkey:
    image: valkey/valkey:8-alpine
    ports: ["127.0.0.1:6381:6379"]
volumes:
  studio_pgdata:
```

Loopback-bound on purpose: reachable from the host, not the LAN.

- [ ] **Step 6: Write the Makefile**

Copy `/Users/sidx/workspace/prot-cellar/Makefile` verbatim, then change the ports to 8002/3002 and every `protcellar` to `daikonstudio`. Keep the awk-based self-documenting `help` target.

- [ ] **Step 7: Run tests and lint**

Run: `cd backend && uv sync && uv run pytest tests/api/test_health.py -v && uv run ruff check . && uv run lint-imports`
Expected: 2 passed, no lint errors, import contracts pass (vacuously — no modules yet).

- [ ] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/src backend/tests docker-compose.yml Makefile
git commit -m "feat: backend skeleton with health and version endpoints"
```

---

### Task 2: Shared domain primitives

**Files:**
- Create: `backend/src/daikonstudio/domain/shared/{entity,errors,events,pagination,provenance,global_workspace}.py`
- Test: `backend/tests/unit/domain/test_shared_entity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Entity`, `AggregateRoot` (with `version`, `register_event`, `collect_events`, `clear_events`); `DomainError` and subclasses `NotFoundError`, `ConflictError`, `ConcurrencyConflictError`, `ValidationError`, `AuthorizationError`, `DataLockedError`, `GoneError`, `ServiceUnavailableError`; `PageResult[T](items, next_cursor, total_count)`; `Provenance` with `GenerationMethod` StrEnum including `AI_PREDICTED`; `GLOBAL_WORKSPACE_ID`.

- [ ] **Step 1: Copy the primitives from prot-cellar**

Copy these files verbatim from `/Users/sidx/workspace/prot-cellar/backend/src/protcellar/domain/shared/` into `backend/src/daikonstudio/domain/shared/`, changing only the `protcellar` import prefix to `daikonstudio`:

`entity.py`, `errors.py`, `events.py`, `pagination.py`, `provenance.py`, `global_workspace.py`

Do **not** copy `compound_ref.py` or `cross_reference.py` — Phase 1 does not need them.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/unit/domain/test_shared_entity.py
import uuid

from daikonstudio.domain.shared.entity import AggregateRoot, Entity
from daikonstudio.domain.shared.errors import ConcurrencyConflictError, NotFoundError
from daikonstudio.domain.shared.global_workspace import GLOBAL_WORKSPACE_ID


def test_entities_are_equal_by_id():
    shared_id = uuid.uuid4()
    assert Entity(id=shared_id) == Entity(id=shared_id)
    assert Entity() != Entity()


def test_aggregate_collects_and_clears_events():
    aggregate = AggregateRoot()
    assert aggregate.version == 1
    aggregate.register_event(object())
    assert len(aggregate.collect_events()) == 1
    aggregate.clear_events()
    assert aggregate.collect_events() == []


def test_not_found_error_names_the_entity():
    error = NotFoundError("Dataset", "abc")
    assert error.message == "Dataset 'abc' not found"


def test_concurrency_error_is_a_domain_error():
    error = ConcurrencyConflictError("Protocol", "abc")
    assert "modified by another transaction" in error.message


def test_global_workspace_is_the_nil_uuid():
    assert GLOBAL_WORKSPACE_ID == uuid.UUID(int=0)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/domain/test_shared_entity.py -v`
Expected: 5 passed. (This task copies proven code, so the test confirms the copy rather than driving new design.)

- [ ] **Step 4: Verify domain purity**

Run: `cd backend && uv run lint-imports`
Expected: "Domain purity" contract KEPT.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/domain/shared backend/tests/unit/domain
git commit -m "feat(domain): shared entity, error, pagination and provenance primitives"
```

---

### Task 3: Persistence base and Alembic

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/base.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/session.py`
- Create: `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/versions/001_initial.py`
- Test: `backend/tests/integration/test_migrations.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces: `Base`, `EntityModelMixin`, `WorkspaceIdMixin`, `VersionMixin`; `create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]`.

- [ ] **Step 1: Copy the persistence base**

Copy `/Users/sidx/workspace/prot-cellar/backend/src/protcellar/infrastructure/persistence/sqlalchemy/base.py` verbatim. It defines `Base`, `EntityModelMixin` (UUID pk, server-default timestamps), `WorkspaceIdMixin` (indexed `workspace_id`), `VersionMixin` (`version` int).

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/integration/test_migrations.py
import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_migrations_apply_cleanly(migrated_session):
    result = await migrated_session.execute(
        text("SELECT version_num FROM alembic_version")
    )
    assert result.scalar_one() is not None
```

Add the fixture in `backend/tests/conftest.py`:

```python
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer


@pytest_asyncio.fixture(scope="session")
async def migrated_session():
    with PostgresContainer("postgres:16-alpine") as postgres:
        sync_url = postgres.get_connection_url()
        async_url = sync_url.replace("postgresql+psycopg2", "postgresql+asyncpg")
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", sync_url)
        command.upgrade(config, "head")
        engine = create_async_engine(async_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            yield session
        await engine.dispose()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_migrations.py -v`
Expected: FAIL — no `alembic.ini`.

- [ ] **Step 4: Write the session factory**

```python
# backend/src/daikonstudio/infrastructure/persistence/session.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 5: Scaffold Alembic**

Run: `cd backend && uv run alembic init -t async alembic`

Then edit `alembic/env.py` to import `Base` and set `target_metadata = Base.metadata`, and to read `STUDIO_DATABASE_URL` from the environment when `sqlalchemy.url` is not set on the config. Generate the empty baseline:

Run: `cd backend && uv run alembic revision -m "initial" --rev-id 001`

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/integration/test_migrations.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/alembic.ini backend/alembic backend/src/daikonstudio/infrastructure/persistence backend/tests
git commit -m "feat(infra): SQLAlchemy base mixins, session factory and Alembic baseline"
```

---

### Task 4: Sentinel auth and workspace guards

**Files:**
- Create: `backend/src/daikonstudio/application/auth.py`
- Create: `backend/src/daikonstudio/infrastructure/sentinel/auth.py`
- Create: `backend/src/daikonstudio/interface/dependencies/_core.py`
- Modify: `backend/src/daikonstudio/interface/app.py`
- Create: `backend/tests/fakes/auth.py`
- Test: `backend/tests/api/test_auth_guards.py`

**Interfaces:**
- Consumes: `DomainError` subclasses (Task 2).
- Produces: `AuthContext` Protocol with read-only properties `user_id: UUID`, `workspace_id: UUID`, `workspace_role: str`; guards `require_authenticated(auth)`, `require_editor(auth)`, `require_admin(auth)`, `require_same_workspace(auth, workspace_id)`; FastAPI dependency `get_auth`; `FakeAuth` test double.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_auth_guards.py
import uuid

import pytest

from daikonstudio.application.auth import (
    require_admin,
    require_editor,
    require_same_workspace,
)
from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError
from tests.fakes.auth import FakeAuth


def test_editor_guard_rejects_viewer():
    with pytest.raises(AuthorizationError):
        require_editor(FakeAuth(workspace_role="viewer"))


def test_editor_guard_allows_editor_and_admin():
    require_editor(FakeAuth(workspace_role="editor"))
    require_editor(FakeAuth(workspace_role="admin"))


def test_admin_guard_rejects_editor():
    with pytest.raises(AuthorizationError):
        require_admin(FakeAuth(workspace_role="editor"))


def test_cross_workspace_access_raises_not_found_not_forbidden():
    """403 would leak the existence of another workspace's resource."""
    auth = FakeAuth(workspace_id=uuid.uuid4())
    with pytest.raises(NotFoundError):
        require_same_workspace(auth, uuid.uuid4(), entity_type="Dataset")


def test_system_calls_bypass_role_guards():
    """auth=None is the worker calling a use case; roles do not apply."""
    require_editor(None)
    require_admin(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_auth_guards.py -v`
Expected: FAIL with `ModuleNotFoundError: daikonstudio.application.auth`

- [ ] **Step 3: Write the guards**

```python
# backend/src/daikonstudio/application/auth.py
"""Workspace guards. Structural typing so the app never imports Sentinel SDK types."""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError

_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}


@runtime_checkable
class AuthContext(Protocol):
    """Structural match for the SDK's RequestAuth without importing it.

    Members are declared as read-only properties, not plain variables: RequestAuth
    implements them as @property, and mypy treats a plain variable in a Protocol as
    read-write, so a read-only property would not satisfy it.

    There is deliberately no `actions` member. RequestAuth has no such attribute —
    action checks go through its async `check_action()`, a different shape entirely.
    No guard here needs it.
    """

    @property
    def user_id(self) -> uuid.UUID: ...

    @property
    def workspace_id(self) -> uuid.UUID: ...

    @property
    def workspace_role(self) -> str: ...


def require_authenticated(auth: AuthContext | None) -> None:
    if auth is None:
        raise AuthorizationError("Authentication required")


def _require_rank(auth: AuthContext | None, minimum: str) -> None:
    if auth is None:  # system/worker call
        return
    if _ROLE_RANK.get(auth.workspace_role, -1) < _ROLE_RANK[minimum]:
        raise AuthorizationError(f"Requires {minimum} role or higher")


def require_editor(auth: AuthContext | None) -> None:
    _require_rank(auth, "editor")


def require_admin(auth: AuthContext | None) -> None:
    _require_rank(auth, "admin")


def require_same_workspace(
    auth: AuthContext | None, workspace_id: uuid.UUID, *, entity_type: str
) -> None:
    if auth is None:
        return
    if auth.workspace_id != workspace_id:
        # NotFound, not Authorization: 403 would confirm the resource exists.
        raise NotFoundError(entity_type)
```

- [ ] **Step 4: Write the fake**

```python
# backend/tests/fakes/auth.py
import uuid
from dataclasses import dataclass, field


@dataclass
class FakeAuth:
    """Plain attributes satisfy the read-only-property Protocol; that direction is fine."""

    user_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_role: str = "editor"
```

- [ ] **Step 5: Wire Sentinel**

```python
# backend/src/daikonstudio/infrastructure/sentinel/auth.py
from sentinel_auth import Sentinel

from daikonstudio.settings import Settings

SERVICE_ACTIONS = [
    "studio:read",
    "studio:write",
    "studio:train",
    "studio:publish",
    "studio:admin_config",
]


def build_sentinel(settings: Settings) -> Sentinel:
    # ponytail: actions are NOT passed to the constructor. The SDK lifespan would
    # register them synchronously and treat failure as fatal, turning rarely-changing
    # housekeeping into a boot blocker. Registered best-effort in app lifespan instead.
    return Sentinel(
        base_url=settings.sentinel_url,
        service_name=settings.service_name,
        service_key=settings.sentinel_service_key,
        mode="authz",
        idp_jwks_url=settings.idp_jwks_url,
        idp_audience=settings.idp_audience,
        idp_issuer=settings.idp_issuer,
        cache_ttl=120,
    )
```

In `app.py`, add the corresponding `Settings` fields, then inside `create_app`:

```python
    sentinel = build_sentinel(settings)
    sentinel.protect(app, exclude_paths=["/health", "/version", "/docs", "/openapi.json"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

`sentinel.protect()` **must** be added before `CORSMiddleware` — Starlette middleware is LIFO, and the reverse order ships 401s without CORS headers.

In `interface/dependencies/_core.py`, `get_auth` returns the Sentinel `request.state.user`; when `STUDIO_SENTINEL_SERVICE_KEY` is unset it is replaced by a dependency that raises `ServiceUnavailableError` — a reject-all stub, never a bypass.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_auth_guards.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/src/daikonstudio/application/auth.py backend/src/daikonstudio/infrastructure/sentinel backend/src/daikonstudio/interface backend/tests
git commit -m "feat(auth): Sentinel authz wiring and workspace guards"
```

---

### Task 5: Blob storage port

**Files:**
- Create: `backend/src/daikonstudio/application/ports/blob_store.py`
- Create: `backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py`
- Test: `backend/tests/integration/test_blob_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BlobStore` Protocol with `put_bytes(key: str, data: bytes) -> str`, `get_bytes(key: str) -> bytes`, `exists(key: str) -> bool`, `delete(key: str) -> None`; `FsspecBlobStore(base_url: str)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/integration/test_blob_store.py
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore


def test_put_get_roundtrip(tmp_path):
    store = FsspecBlobStore(f"file://{tmp_path}")
    uri = store.put_bytes("ws/datasets/abc/snapshot.parquet", b"payload")
    assert store.get_bytes("ws/datasets/abc/snapshot.parquet") == b"payload"
    assert uri.endswith("ws/datasets/abc/snapshot.parquet")


def test_exists_and_delete(tmp_path):
    store = FsspecBlobStore(f"file://{tmp_path}")
    assert store.exists("missing") is False
    store.put_bytes("present", b"x")
    assert store.exists("present") is True
    store.delete("present")
    assert store.exists("present") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_blob_store.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the port and implementation**

```python
# backend/src/daikonstudio/application/ports/blob_store.py
from typing import Protocol


class BlobStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
```

```python
# backend/src/daikonstudio/infrastructure/storage/fsspec_blob_store.py
"""One implementation, driven by BLOB_BASE_URL. file:// in dev, s3:// in prod."""

import fsspec


class FsspecBlobStore:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._fs, _ = fsspec.core.url_to_fs(self._base)

    def _path(self, key: str) -> str:
        return f"{self._base}/{key.lstrip('/')}"

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        self._fs.makedirs(self._fs._parent(path), exist_ok=True)
        with self._fs.open(path, "wb") as handle:
            handle.write(data)
        return path

    def get_bytes(self, key: str) -> bytes:
        with self._fs.open(self._path(key), "rb") as handle:
            return handle.read()

    def exists(self, key: str) -> bool:
        return bool(self._fs.exists(self._path(key)))

    def delete(self, key: str) -> None:
        self._fs.rm(self._path(key))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/integration/test_blob_store.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/ports backend/src/daikonstudio/infrastructure/storage backend/tests/integration/test_blob_store.py
git commit -m "feat(infra): fsspec blob store behind a BlobStore port"
```

---

### Task 6: Chemistry utilities

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/chem/{canonicalize,featurize,scaffold,similarity}.py`
- Test: `backend/tests/unit/chem/test_chem_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `canonicalize(smiles: str) -> str | None`; `has_multiple_components(smiles: str) -> bool`; `ecfp4(smiles_list: list[str]) -> np.ndarray` (shape `(n, 2048)`, uint8); `murcko_scaffold(smiles: str) -> str`; `nearest_neighbour_tanimoto(query: list[str], reference: list[str]) -> np.ndarray` (shape `(len(query),)`, float, max similarity to any reference).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/chem/test_chem_utils.py
import numpy as np

from daikonstudio.infrastructure.chem.canonicalize import canonicalize, has_multiple_components
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.chem.scaffold import murcko_scaffold
from daikonstudio.infrastructure.chem.similarity import nearest_neighbour_tanimoto


def test_canonicalize_normalises_equivalent_smiles():
    assert canonicalize("C1=CC=CC=C1") == canonicalize("c1ccccc1")


def test_canonicalize_returns_none_for_garbage():
    assert canonicalize("not-a-molecule") is None


def test_salts_are_detected_as_multi_component():
    assert has_multiple_components("CC(=O)O.[Na+]") is True
    assert has_multiple_components("CCO") is False


def test_ecfp4_shape_and_determinism():
    first = ecfp4(["CCO", "c1ccccc1"])
    second = ecfp4(["CCO", "c1ccccc1"])
    assert first.shape == (2, 2048)
    assert np.array_equal(first, second)


def test_murcko_scaffold_strips_side_chains():
    """Toluene and benzene share the benzene scaffold."""
    assert murcko_scaffold("Cc1ccccc1") == murcko_scaffold("c1ccccc1")


def test_nearest_neighbour_similarity_is_one_for_exact_match():
    scores = nearest_neighbour_tanimoto(["CCO"], ["CCO", "c1ccccc1"])
    assert scores.shape == (1,)
    assert scores[0] == 1.0


def test_nearest_neighbour_similarity_is_low_for_dissimilar():
    scores = nearest_neighbour_tanimoto(["CCCCCCCCCC"], ["c1ccc2ccccc2c1"])
    assert scores[0] < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/chem -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the implementations**

```python
# backend/src/daikonstudio/infrastructure/chem/canonicalize.py
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # invalid input is expected and reported, not logged


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def has_multiple_components(smiles: str) -> bool:
    return "." in (canonicalize(smiles) or "")
```

```python
# backend/src/daikonstudio/infrastructure/chem/featurize.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(smiles_list: list[str]) -> np.ndarray:
    """ECFP4 (Morgan radius 2, 2048 bits). Invalid SMILES yield an all-zero row."""
    rows = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rows[index] = np.frombuffer(
                _GENERATOR.GetFingerprintAsNumPy(mol).tobytes(), dtype=np.uint8
            )[:2048]
    return rows
```

```python
# backend/src/daikonstudio/infrastructure/chem/scaffold.py
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold. Returns "" for invalid input or acyclic molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
```

```python
# backend/src/daikonstudio/infrastructure/chem/similarity.py
import numpy as np

from daikonstudio.infrastructure.chem.featurize import ecfp4


def nearest_neighbour_tanimoto(query: list[str], reference: list[str]) -> np.ndarray:
    """Max Tanimoto from each query molecule to any reference molecule."""
    if not reference:
        return np.zeros(len(query))
    q = ecfp4(query).astype(np.float32)
    r = ecfp4(reference).astype(np.float32)
    intersection = q @ r.T
    union = q.sum(axis=1)[:, None] + r.sum(axis=1)[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    return similarity.max(axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/chem -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/infrastructure/chem backend/tests/unit/chem
git commit -m "feat(chem): canonicalization, ECFP4, Murcko scaffolds and Tanimoto similarity"
```

---

### Task 7: Engine contract and registry

**Files:**
- Create: `backend/src/daikonstudio/application/engines/{manifest,protocol,context,registry}.py`
- Test: `backend/tests/unit/engines/test_engine_contract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately dependency-free so it stays portable when engines move out of process).
- Produces:
  - `TaskType` StrEnum: `REGRESSION`, `BINARY_CLASSIFICATION`
  - `ConditionType` StrEnum: `STRING`, `INTEGER`, `NUMBER`, `ENUM`, `BOOL`
  - `ConditionSpec(key, label, type, required=False, default=None, minimum=None, maximum=None, options=(), help=None)`
  - `EngineManifest(id, version, name, description, tasks, conditions=(), is_baseline=False)`
  - `TrainContext(frame, task, structure_column, target_column, conditions, seed)`, `TrainResult(artifact, metrics)`
  - `PredictContext(frame, artifact, conditions)`
  - `Engine` Protocol: `manifest() -> EngineManifest` (staticmethod), `train(ctx) -> TrainResult`, `predict(ctx) -> pl.DataFrame`
  - `EngineRegistry.get(engine_id) -> Engine`, `.manifests() -> list[EngineManifest]`, `.baseline() -> Engine`
  - `validate_conditions(manifest, supplied) -> dict[str, object]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/engines/test_engine_contract.py
import polars as pl
import pytest

from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError

MANIFEST = EngineManifest(
    id="fake",
    version="1.0.0",
    name="Fake",
    description="test double",
    tasks=(TaskType.REGRESSION,),
    conditions=(
        ConditionSpec(
            key="n_estimators",
            label="Trees",
            type=ConditionType.INTEGER,
            default=100,
            minimum=1,
            maximum=1000,
        ),
    ),
)


class FakeEngine:
    @staticmethod
    def manifest() -> EngineManifest:
        return MANIFEST

    def train(self, ctx):
        return None

    def predict(self, ctx):
        return pl.DataFrame({"row_id": [], "value": []})


def test_conditions_fill_in_defaults():
    assert validate_conditions(MANIFEST, {}) == {"n_estimators": 100}


def test_conditions_reject_out_of_range():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": 5000})


def test_conditions_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        validate_conditions(MANIFEST, {"learning_rate": 0.1})


def test_registry_returns_registered_engine():
    registry = EngineRegistry({"fake": FakeEngine()})
    assert registry.get("fake").manifest().id == "fake"
    assert [m.id for m in registry.manifests()] == ["fake"]


def test_registry_raises_for_unknown_engine():
    with pytest.raises(UnknownEngineError):
        EngineRegistry({}).get("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/engines -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the manifest module**

```python
# backend/src/daikonstudio/application/engines/manifest.py
"""The Engine contract's declarative half.

Constrained exactly as far as prot-cellar's ParamField: enough that the frontend
renders the condition form directly, and no further. The manifest is deliberately
free of infrastructure imports so this envelope serializes to HTTP unchanged when
engines move out of process in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskType(StrEnum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


class ConditionType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    ENUM = "enum"
    BOOL = "bool"


@dataclass(frozen=True, kw_only=True)
class ConditionSpec:
    key: str
    label: str
    type: ConditionType
    required: bool = False
    default: object | None = None
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()
    help: str | None = None


@dataclass(frozen=True, kw_only=True)
class EngineManifest:
    id: str
    version: str
    name: str
    description: str
    tasks: tuple[TaskType, ...]
    conditions: tuple[ConditionSpec, ...] = ()
    is_baseline: bool = False


def validate_conditions(
    manifest: EngineManifest, supplied: dict[str, object]
) -> dict[str, object]:
    """Fill defaults, reject unknown keys and out-of-range values."""
    known = {c.key: c for c in manifest.conditions}
    unknown = set(supplied) - set(known)
    if unknown:
        raise ValueError(f"unknown conditions for {manifest.id}: {sorted(unknown)}")

    resolved: dict[str, object] = {}
    for key, spec in known.items():
        if key in supplied:
            value = supplied[key]
        elif spec.default is not None:
            value = spec.default
        elif spec.required:
            raise ValueError(f"{key} is required for {manifest.id}")
        else:
            continue

        if spec.minimum is not None and float(value) < spec.minimum:  # type: ignore[arg-type]
            raise ValueError(f"{key} below minimum {spec.minimum}")
        if spec.maximum is not None and float(value) > spec.maximum:  # type: ignore[arg-type]
            raise ValueError(f"{key} above maximum {spec.maximum}")
        if spec.options and value not in spec.options:
            raise ValueError(f"{key} must be one of {spec.options}")
        resolved[key] = value
    return resolved
```

- [ ] **Step 4: Write the context, protocol and registry**

```python
# backend/src/daikonstudio/application/engines/context.py
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from daikonstudio.application.engines.manifest import TaskType


@dataclass(frozen=True, kw_only=True)
class TrainContext:
    """`frame` carries the dataset columns plus a `split` column of train/validation/test.

    `task` is passed explicitly and is authoritative. An engine must NEVER infer
    regression-vs-classification from the target values: a regression target whose
    values happen to all be 0.0 or 1.0 would silently train a classifier. The
    Dataset's TargetSpec is the only source of truth for what is being predicted.
    """

    frame: pl.DataFrame
    task: TaskType
    structure_column: str
    target_column: str
    conditions: dict[str, object]
    seed: int


@dataclass(frozen=True, kw_only=True)
class TrainResult:
    artifact: bytes
    metrics: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class PredictContext:
    frame: pl.DataFrame
    structure_column: str
    artifact: bytes
    conditions: dict[str, object]
```

```python
# backend/src/daikonstudio/application/engines/protocol.py
from __future__ import annotations

from typing import Protocol

import polars as pl

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import EngineManifest


class Engine(Protocol):
    """Structural — an engine matches this shape; no base class.

    train/predict are SYNCHRONOUS on purpose. They are CPU-bound and would block the
    event loop; the worker calls them via asyncio.to_thread. Engine authors write
    plain sync code and never think about async.
    """

    @staticmethod
    def manifest() -> EngineManifest: ...

    def train(self, ctx: TrainContext) -> TrainResult: ...

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        """Returns columns: row_id (int), value (float), uncertainty (float | null)."""
        ...
```

```python
# backend/src/daikonstudio/application/engines/registry.py
from __future__ import annotations

from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine


class UnknownEngineError(KeyError):
    pass


class EngineRegistry:
    def __init__(self, engines: dict[str, Engine]) -> None:
        self._engines = engines

    def get(self, engine_id: str) -> Engine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise UnknownEngineError(engine_id) from exc

    def manifests(self) -> list[EngineManifest]:
        return [engine.manifest() for engine in self._engines.values()]

    def baseline(self) -> Engine:
        for engine in self._engines.values():
            if engine.manifest().is_baseline:
                return engine
        raise UnknownEngineError("no baseline engine registered")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/engines -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/application/engines backend/tests/unit/engines
git commit -m "feat(engines): manifest, condition validation, Engine protocol and registry"
```

---

### Task 8: The two ECFP4 engines

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/engines/{ecfp4_xgboost,ecfp4_randomforest}.py`
- Test: `backend/tests/unit/engines/test_ecfp4_engines.py`

**Interfaces:**
- Consumes: `Engine`, `EngineManifest`, `TrainContext`, `TrainResult`, `PredictContext` (Task 7); `ecfp4` (Task 6).
- Produces: `Ecfp4XGBoost` (id `ecfp4-xgboost`), `Ecfp4RandomForest` (id `ecfp4-randomforest`, `is_baseline=True`). Both support both `TaskType`s.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/engines/test_ecfp4_engines.py
import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
          "CCN", "CCCN", "CCCCN", "c1ccncc1", "Cc1ccncc1", "CCc1ccncc1"]
VALUES = [1.0, 1.2, 1.4, 5.0, 5.2, 5.4, 2.0, 2.2, 2.4, 6.0, 6.2, 6.4]
SPLIT = ["train"] * 8 + ["test"] * 4


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": SMILES, "y": VALUES, "split": SPLIT})


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_train_returns_artifact_and_metrics(engine):
    ctx = TrainContext(
        frame=frame(), task=TaskType.REGRESSION, structure_column="smiles",
        target_column="y", conditions={}, seed=42,
    )
    result = engine.train(ctx)
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0
    assert "rmse" in result.metrics


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_predict_returns_one_row_per_input(engine):
    ctx = TrainContext(
        frame=frame(), task=TaskType.REGRESSION, structure_column="smiles",
        target_column="y", conditions={}, seed=42,
    )
    artifact = engine.train(ctx).artifact
    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]}),
            structure_column="smiles", artifact=artifact, conditions={},
        )
    )
    assert predictions.height == 2
    assert set(predictions.columns) == {"row_id", "value", "uncertainty"}


def test_training_is_reproducible_from_the_seed():
    ctx = TrainContext(
        frame=frame(), task=TaskType.REGRESSION, structure_column="smiles",
        target_column="y", conditions={}, seed=42,
    )
    first = Ecfp4RandomForest().train(ctx).metrics["rmse"]
    second = Ecfp4RandomForest().train(ctx).metrics["rmse"]
    assert first == second


def test_a_regression_target_of_only_zeros_and_ones_still_trains_a_regressor():
    """Guards the sniffing bug: task comes from TargetSpec, never from the values."""
    binary_looking = pl.DataFrame({
        "smiles": SMILES, "y": [0.0, 1.0] * 6, "split": SPLIT,
    })
    ctx = TrainContext(
        frame=binary_looking, task=TaskType.REGRESSION, structure_column="smiles",
        target_column="y", conditions={}, seed=42,
    )
    assert "rmse" in Ecfp4RandomForest().train(ctx).metrics


def test_random_forest_is_flagged_as_the_baseline():
    assert Ecfp4RandomForest.manifest().is_baseline is True
    assert Ecfp4XGBoost.manifest().is_baseline is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/engines/test_ecfp4_engines.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the RandomForest baseline**

```python
# backend/src/daikonstudio/infrastructure/engines/ecfp4_randomforest.py
"""ECFP4 + RandomForest. The mandatory baseline every Scorecard compares against.

In the Polaris ADMET competition, fingerprint baselines placed around 20th of 66
teams. A user whose engine cannot beat this needs to know on the first screen.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.infrastructure.chem.featurize import ecfp4

_MANIFEST = EngineManifest(
    id="ecfp4-randomforest",
    version="1.0.0",
    name="ECFP4 + Random Forest",
    description="Morgan fingerprints with a random forest. Fast, robust, the baseline.",
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="n_estimators", label="Trees", type=ConditionType.INTEGER,
            default=500, minimum=10, maximum=2000,
            help="More trees is steadier and slower. 500 is a good default.",
        ),
    ),
    is_baseline=True,
)


class Ecfp4RandomForest:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        x_train = ecfp4(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        model_class = RandomForestClassifier if is_classification else RandomForestRegressor
        model = model_class(
            n_estimators=int(conditions["n_estimators"]), random_state=ctx.seed, n_jobs=-1
        )
        model.fit(x_train, y_train)

        buffer = io.BytesIO()
        joblib.dump({"model": model, "is_classification": is_classification}, buffer)
        return TrainResult(
            artifact=buffer.getvalue(),
            metrics=_score(model, test_rows, ctx, is_classification),
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
```

Add a shared `backend/src/daikonstudio/infrastructure/engines/_scoring.py` holding `_score` (RMSE/MAE/R² for regression; MCC/balanced accuracy/AUROC/AUPRC for classification) and `_predict_with_tree_ensemble`, which returns per-row `uncertainty` as the standard deviation across estimators for regression and `abs(p - 0.5)` inverted for classification.

- [ ] **Step 4: Write the XGBoost engine**

Same structure with `XGBRegressor`/`XGBClassifier`, id `ecfp4-xgboost`, `is_baseline=False`, and conditions `n_estimators` (default 400), `max_depth` (default 6, 1–20), `learning_rate` (default 0.1, 0.001–1.0). Set `random_state=ctx.seed` and `n_jobs=-1`. Uncertainty for XGBoost is null for regression (no ensemble spread available) — return a null column rather than a fabricated number.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/engines/test_ecfp4_engines.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/infrastructure/engines backend/tests/unit/engines/test_ecfp4_engines.py
git commit -m "feat(engines): ECFP4 XGBoost and RandomForest baseline engines"
```

---

### Task 9: Dataset validation and deduplication

**Files:**
- Create: `backend/src/daikonstudio/domain/data/{target,validation}.py`
- Create: `backend/src/daikonstudio/application/data/prepare_frame.py`
- Test: `backend/tests/unit/data/test_validation.py`

**Interfaces:**
- Consumes: `canonicalize`, `has_multiple_components` (Task 6).
- Produces:
  - `TargetKind` StrEnum: `NUMERIC`, `BINARY`; `Direction` StrEnum: `HIGH`, `LOW`
  - `TargetSpec(column, kind, unit=None, direction=None)`
  - `InvalidRow(row_number, value, reason)`, `ConflictRow(structure, values)`
  - `ValidationReport(total_rows, valid_rows, invalid, duplicates_collapsed, conflicting, salts_flagged, duplicate_spread)`
  - `prepare_frame(frame, structure_column, target: TargetSpec) -> tuple[pl.DataFrame, ValidationReport]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/data/test_validation.py
import polars as pl
import pytest

from daikonstudio.application.data.prepare_frame import prepare_frame
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec

NUMERIC = TargetSpec(column="y", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW)
BINARY = TargetSpec(column="y", kind=TargetKind.BINARY)


def test_invalid_structures_are_reported_with_row_numbers():
    frame = pl.DataFrame({"smiles": ["CCO", "not-a-molecule"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC)
    assert prepared.height == 1
    assert report.invalid[0].row_number == 2
    assert "invalid structure" in report.invalid[0].reason


def test_numeric_duplicates_are_averaged_and_spread_retained():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC", "c1ccccc1"], "y": [1.0, 3.0, 9.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC)
    assert prepared.height == 2
    assert prepared.filter(pl.col("smiles") == "CCO")["y"].item() == 2.0
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == pytest.approx(2.0)  # |1.0 - 3.0|


def test_conflicting_binary_duplicates_are_rejected_not_voted():
    """A compound labelled both active and inactive is a data problem to decide about."""
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [0, 1]})
    prepared, report = prepare_frame(frame, "smiles", BINARY)
    assert prepared.height == 0
    assert report.conflicting[0].values == [0, 1]


def test_agreeing_binary_duplicates_collapse_silently():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [1, 1]})
    prepared, report = prepare_frame(frame, "smiles", BINARY)
    assert prepared.height == 1
    assert report.conflicting == []
    assert report.duplicates_collapsed == 1


def test_salts_are_flagged_but_kept():
    frame = pl.DataFrame({"smiles": ["CC(=O)O.[Na+]", "CCO"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC)
    assert prepared.height == 2
    assert report.salts_flagged == 1


def test_structures_are_canonicalised_so_equivalent_smiles_deduplicate():
    frame = pl.DataFrame({"smiles": ["C1=CC=CC=C1", "c1ccccc1"], "y": [1.0, 1.0]})
    prepared, _ = prepare_frame(frame, "smiles", NUMERIC)
    assert prepared.height == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/data -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the domain value objects**

```python
# backend/src/daikonstudio/domain/data/target.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetKind(StrEnum):
    NUMERIC = "numeric"
    BINARY = "binary"


class Direction(StrEnum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, kw_only=True)
class TargetSpec:
    """What the scientist is predicting, and in what units.

    Readouts on a trained Protocol are derived from this, which is the mechanism by
    which a predicted IC50 arrives in the same unit and direction as a measured one.
    """

    column: str
    kind: TargetKind
    unit: str | None = None
    direction: Direction | None = None
```

```python
# backend/src/daikonstudio/domain/data/validation.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class InvalidRow:
    row_number: int
    value: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ConflictRow:
    structure: str
    values: list[int]


@dataclass(frozen=True, kw_only=True)
class ValidationReport:
    total_rows: int
    valid_rows: int
    invalid: list[InvalidRow] = field(default_factory=list)
    conflicting: list[ConflictRow] = field(default_factory=list)
    duplicates_collapsed: int = 0
    salts_flagged: int = 0
    duplicate_spread: float | None = None
```

- [ ] **Step 4: Write `prepare_frame`**

Implement in `backend/src/daikonstudio/application/data/prepare_frame.py`, in this order:

1. Canonicalize the structure column; rows returning `None` become `InvalidRow(row_number=index+1, reason="invalid structure")` and are dropped.
2. Count multi-component structures into `salts_flagged` (flagged, not dropped).
3. Group by canonical structure. For `NUMERIC`, average the target and accumulate `max - min` per duplicate group into `duplicate_spread` (mean of group spreads). For `BINARY`, groups whose values disagree become `ConflictRow` and are dropped; agreeing groups collapse.
4. `duplicates_collapsed` counts rows removed by grouping.

Deduplicating **before** splitting eliminates train/test structure leakage by construction rather than warning about it afterwards.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/data -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/domain/data backend/src/daikonstudio/application/data backend/tests/unit/data
git commit -m "feat(data): structure validation, canonicalization and duplicate collapsing"
```

---

### Task 10: Splits and the frozen snapshot

**Files:**
- Create: `backend/src/daikonstudio/domain/data/split.py`
- Create: `backend/src/daikonstudio/application/data/{assign_split,snapshot}.py`
- Test: `backend/tests/unit/data/test_split.py`

**Interfaces:**
- Consumes: `murcko_scaffold` (Task 6); `BlobStore` (Task 5).
- Produces:
  - `SplitStrategy` StrEnum: `RANDOM`, `SCAFFOLD`
  - `SplitSpec(strategy, seed, fractions=(0.8, 0.1, 0.1))`
  - `assign_split(frame, structure_column, spec) -> pl.DataFrame` (adds a `split` column of `train`/`validation`/`test`)
  - `write_snapshot(store, workspace_id, dataset_id, frame) -> tuple[str, str]` returning `(uri, content_hash)`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/data/test_split.py
import polars as pl

from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
          "CCN", "CCCN", "c1ccncc1", "Cc1ccncc1"]


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": SMILES})


def test_random_split_is_reproducible_from_the_seed():
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=7)
    first = assign_split(frame(), "smiles", spec)["split"].to_list()
    second = assign_split(frame(), "smiles", spec)["split"].to_list()
    assert first == second


def test_random_split_respects_the_fractions():
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=7, fractions=(0.8, 0.1, 0.1))
    counts = assign_split(frame(), "smiles", spec)["split"].value_counts().to_dict()
    assert dict(zip(counts["split"], counts["count"]))["train"] == 8


def test_scaffold_split_keeps_a_scaffold_within_one_partition():
    """Benzene, toluene and ethylbenzene share a scaffold and must not straddle splits."""
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    result = assign_split(frame(), "smiles", spec)
    benzenes = result.filter(pl.col("smiles").is_in(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]))
    assert benzenes["split"].n_unique() == 1


def test_every_row_receives_a_partition():
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    result = assign_split(frame(), "smiles", spec)
    assert result["split"].null_count() == 0
    assert set(result["split"].unique()) <= {"train", "validation", "test"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/data/test_split.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `SplitSpec`**

```python
# backend/src/daikonstudio/domain/data/split.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SplitStrategy(StrEnum):
    RANDOM = "random"
    SCAFFOLD = "scaffold"
    # ponytail: Butina, UMAP-cluster and temporal splits land in Phase 2. Two
    # strategies is the minimum that makes the optimism gap on the Scorecard real.


@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    """A named, visible scientific choice — never a silent 80/10/10."""

    strategy: SplitStrategy
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
```

- [ ] **Step 4: Write `assign_split` and `write_snapshot`**

`assign_split` for `RANDOM`: shuffle row indices with `numpy.random.default_rng(spec.seed)`, slice by the fractions. For `SCAFFOLD`: compute `murcko_scaffold` per row, group rows by scaffold, sort groups by descending size (largest scaffolds into train, which is the standard deterministic scaffold split), then fill train, validation and test to their target sizes group by group so a scaffold never straddles partitions.

`write_snapshot` serializes the frame to Parquet bytes, computes `content_hash = sha256(parquet_bytes)`, and writes to `{workspace_id}/datasets/{dataset_id}/snapshot.parquet` via the `BlobStore`, returning `(uri, content_hash)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/data/test_split.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/domain/data/split.py backend/src/daikonstudio/application/data backend/tests/unit/data/test_split.py
git commit -m "feat(data): random and scaffold splits with content-addressed Parquet snapshots"
```

---

### Task 11: Dataset aggregate, persistence and routes

**Files:**
- Create: `backend/src/daikonstudio/domain/data/dataset.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/{models,repository}.py`
- Create: `backend/src/daikonstudio/application/data/{create_dataset,get_dataset,list_datasets}.py`
- Create: `backend/src/daikonstudio/interface/routes/datasets.py`
- Create: `backend/alembic/versions/002_datasets.py`
- Test: `backend/tests/api/test_datasets.py`

**Interfaces:**
- Consumes: `prepare_frame` (Task 9), `assign_split`/`write_snapshot` (Task 10), guards (Task 4), `BlobStore` (Task 5).
- Produces: `Dataset` aggregate (`id`, `workspace_id`, `name`, `target: TargetSpec`, `split: SplitSpec`, `content_hash`, `snapshot_uri`, `row_count`, `validation_report`, `version`); routes `POST /api/v1/datasets/uploads`, `POST /api/v1/datasets`, `GET /api/v1/datasets`, `GET /api/v1/datasets/{id}`; and:

```python
@dataclass(frozen=True, kw_only=True)
class CreateDatasetCommand:
    name: str
    upload_ref: str
    structure_column: str
    target: TargetSpec
    split: SplitSpec


class CreateDataset:
    def __call__(
        self, command: CreateDatasetCommand, auth: AuthContext | None
    ) -> Result[Dataset, DomainError]: ...
```

- [ ] **Step 1: Write the failing API test**

```python
# backend/tests/api/test_datasets.py
import io


def test_create_dataset_returns_201_with_validation_report(client, csv_upload):
    upload_ref = csv_upload(b"smiles,y\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n")
    response = client.post("/api/v1/datasets", json={
        "name": "solubility", "upload_ref": upload_ref,
        "structure_column": "smiles",
        "target": {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
        "split": {"strategy": "scaffold", "seed": 42},
    })
    assert response.status_code == 201
    body = response.json()
    assert body["row_count"] == 4
    assert body["validation_report"]["invalid"] == []
    assert body["content_hash"]


def test_create_dataset_rejects_a_frame_with_no_valid_structures(client, csv_upload):
    upload_ref = csv_upload(b"smiles,y\nnope,1.0\nalso-nope,2.0\n")
    response = client.post("/api/v1/datasets", json={
        "name": "broken", "upload_ref": upload_ref, "structure_column": "smiles",
        "target": {"column": "y", "kind": "numeric"},
        "split": {"strategy": "random", "seed": 1},
    })
    assert response.status_code == 422
    assert len(response.json()["detail"]["invalid"]) == 2


def test_dataset_is_scoped_to_the_callers_workspace(client, other_workspace_client, csv_upload):
    upload_ref = csv_upload(b"smiles,y\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n")
    created = client.post("/api/v1/datasets", json={
        "name": "private", "upload_ref": upload_ref, "structure_column": "smiles",
        "target": {"column": "y", "kind": "numeric"},
        "split": {"strategy": "random", "seed": 1},
    }).json()
    assert other_workspace_client.get(f"/api/v1/datasets/{created['id']}").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_datasets.py -v`
Expected: FAIL — route not registered (404 on POST).

- [ ] **Step 3: Write the aggregate and the ORM model**

`Dataset(AggregateRoot)` holds the fields listed under Interfaces and is immutable after construction — there is no mutating method. `DatasetModel(Base, EntityModelMixin, WorkspaceIdMixin, VersionMixin)` maps it with `target` and `split` and `validation_report` as JSONB, plus a unique index on `(workspace_id, content_hash)` so the same frozen data is never stored twice.

- [ ] **Step 4: Write `CreateDataset`**

```python
class CreateDataset:
    def __call__(self, command, auth) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        # ... read upload -> pl.read_csv -> prepare_frame -> reject if valid_rows == 0
        # ... assign_split -> write_snapshot -> persist
```

Guards are the first two lines. `workspace_id` comes from `auth.workspace_id`. If `report.valid_rows == 0`, return `Failure(ValidationError("No valid structures", detail=report))` — the route renders the whole report into the 422 body, because the report is the useful part.

- [ ] **Step 5: Write the routes and the migration**

`POST /api/v1/datasets/uploads` accepts multipart, stores the bytes under `{workspace_id}/uploads/{uuid}.csv` and returns `{"upload_ref": ...}`. `POST /api/v1/datasets` returns 201. `GET /api/v1/datasets` returns `PaginatedResponse[DatasetResponse]` using the shared cursor helpers. Generate the migration with `uv run alembic revision --autogenerate -m "datasets" --rev-id 002`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_datasets.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/src/daikonstudio backend/alembic/versions backend/tests/api/test_datasets.py
git commit -m "feat(data): Dataset aggregate, persistence and CRUD routes"
```

---

### Task 12: Protocol aggregate and readout derivation

**Files:**
- Create: `backend/src/daikonstudio/domain/catalog/{protocol,readout}.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/{models,repository}.py`
- Create: `backend/alembic/versions/003_protocols.py`
- Test: `backend/tests/unit/catalog/test_protocol.py`

**Interfaces:**
- Consumes: `TargetSpec`, `TaskType`.
- Produces:
  - `ReadoutType` StrEnum: `NUMERIC`, `PROBABILITY`, `CLASS`
  - `Readout(name, type, unit, direction, description)`
  - `ProtocolStatus` StrEnum: `DRAFT`, `PUBLISHED`
  - `InSilicoProtocol(AggregateRoot)` with `publish()`, `is_locked`, `parent_protocol_id`, `protocol_version`
  - `derive_readouts(target: TargetSpec, task: TaskType) -> tuple[Readout, ...]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/catalog/test_protocol.py
import pytest

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import ReadoutType, derive_readouts
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.shared.errors import DataLockedError


def test_regression_readout_inherits_unit_and_direction_from_the_target():
    target = TargetSpec(column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW)
    readouts = derive_readouts(target, TaskType.REGRESSION)
    assert len(readouts) == 1
    assert readouts[0].type == ReadoutType.NUMERIC
    assert readouts[0].unit == "nM"
    assert readouts[0].direction == Direction.LOW


def test_classification_produces_a_probability_and_a_class_readout():
    target = TargetSpec(column="active", kind=TargetKind.BINARY)
    readouts = derive_readouts(target, TaskType.BINARY_CLASSIFICATION)
    assert [r.type for r in readouts] == [ReadoutType.PROBABILITY, ReadoutType.CLASS]
    assert readouts[0].unit is None


def test_publishing_locks_the_protocol():
    protocol = _draft()
    assert protocol.status == ProtocolStatus.DRAFT
    protocol.publish()
    assert protocol.status == ProtocolStatus.PUBLISHED
    assert protocol.is_locked is True


def test_a_published_protocol_cannot_be_republished():
    protocol = _draft()
    protocol.publish()
    with pytest.raises(DataLockedError):
        protocol.publish()


def test_a_new_version_chains_to_its_parent():
    parent = _draft()
    parent.publish()
    child = parent.new_version(artifact_uri="s3://x/2")
    assert child.parent_protocol_id == parent.id
    assert child.protocol_version == parent.protocol_version + 1
    assert child.status == ProtocolStatus.DRAFT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/catalog -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `derive_readouts`**

```python
# backend/src/daikonstudio/domain/catalog/readout.py
def derive_readouts(target: TargetSpec, task: TaskType) -> tuple[Readout, ...]:
    """An Engine cannot declare its concrete outputs — they depend on what it was
    trained on. Readouts are derived from the Dataset's TargetSpec at training time,
    which is how a predicted IC50 arrives in the same unit as a measured one."""
    if task is TaskType.REGRESSION:
        return (
            Readout(
                name=target.column, type=ReadoutType.NUMERIC,
                unit=target.unit, direction=target.direction,
                description=f"Predicted {target.column}",
            ),
        )
    return (
        Readout(
            name=f"{target.column}_probability", type=ReadoutType.PROBABILITY,
            unit=None, direction=Direction.HIGH,
            description=f"Probability that {target.column} is positive",
        ),
        Readout(
            name=target.column, type=ReadoutType.CLASS,
            unit=None, direction=target.direction,
            description=f"Predicted {target.column} class",
        ),
    )
```

- [ ] **Step 4: Write the aggregate**

`InSilicoProtocol.publish()` raises `DataLockedError` when already published and otherwise sets `status`, `is_locked`, and `published_at`. `new_version(artifact_uri)` returns a fresh DRAFT with `parent_protocol_id=self.id` and `protocol_version=self.protocol_version + 1`. Nothing else mutates after publish — immutability is what makes a Protocol citable.

- [ ] **Step 5: Write the ORM model and migration**

`InSilicoProtocolModel` with `readouts` and `conditions` as JSONB, `parent_protocol_id` a self-referencing nullable UUID, and a `CheckConstraint("status IN ('draft','published')")`. Generate with `--rev-id 003`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/catalog -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/src/daikonstudio/domain/catalog backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog backend/alembic/versions backend/tests/unit/catalog
git commit -m "feat(catalog): Protocol aggregate with derived readouts and publish locking"
```

---

### Task 13: Run aggregate and the arq worker

**Files:**
- Create: `backend/src/daikonstudio/domain/execution/run.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/{models,repository}.py`
- Create: `backend/src/daikonstudio/infrastructure/worker.py`
- Create: `backend/src/daikonstudio/application/execution/enqueue.py`
- Create: `backend/alembic/versions/004_runs.py`
- Test: `backend/tests/integration/test_run_lifecycle.py`

**Interfaces:**
- Consumes: guards (Task 4).
- Produces:
  - `RunKind` StrEnum: `TRAINING`, `PREDICTION`
  - `RunStatus` StrEnum: `PENDING`, `RUNNING`, `READY`, `FAILED`, `CANCELLED`
  - `Run(AggregateRoot)` with `start()`, `report_progress(fraction, phase)`, `succeed(result_uri)`, `fail(message)`, `cancel()`
  - `JobEnqueuer` Protocol with `enqueue(run_id: UUID) -> None`; `ArqEnqueuer` and `InlineEnqueuer` (the test/dev implementation that runs the job immediately)
  - `compute_cache_key(**parts) -> str`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/integration/test_run_lifecycle.py
import uuid

import pytest

from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import ConflictError


def _pending() -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="deadbeef",
    )


def test_run_starts_pending():
    run = _pending()
    assert run.status is RunStatus.PENDING
    assert run.progress == 0.0


def test_lifecycle_transitions_to_ready():
    run = _pending()
    run.start()
    assert run.status is RunStatus.RUNNING
    run.report_progress(0.5, phase="training baseline")
    assert run.progress == 0.5 and run.phase == "training baseline"
    run.succeed("s3://results.parquet")
    assert run.status is RunStatus.READY and run.result_uri.endswith(".parquet")


def test_failure_records_the_message():
    run = _pending()
    run.start()
    run.fail("engine raised ValueError")
    assert run.status is RunStatus.FAILED
    assert run.error_message == "engine raised ValueError"


def test_a_terminal_run_cannot_restart():
    run = _pending()
    run.start()
    run.succeed("s3://x")
    with pytest.raises(ConflictError):
        run.start()


def test_cancel_is_allowed_from_pending_and_running_only():
    run = _pending()
    run.cancel()
    assert run.status is RunStatus.CANCELLED
    with pytest.raises(ConflictError):
        run.cancel()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_run_lifecycle.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the Run aggregate**

Statuses exactly as the siblings use them: `pending/running/ready/failed/cancelled`, enforced by a `CheckConstraint` on the model. `_TERMINAL = {READY, FAILED, CANCELLED}`; any transition out of a terminal state raises `ConflictError`.

- [ ] **Step 4: Write the worker**

```python
# backend/src/daikonstudio/infrastructure/worker.py
"""arq worker. One task, dispatching on Run.kind — prot-cellar's run_import shape."""

async def run_job(ctx, run_id: uuid.UUID) -> None:
    run = await _load(ctx, run_id)
    run.start()
    await _save(ctx, run)
    try:
        handler = _HANDLERS[run.kind]
        result_uri = await handler(ctx, run)
        run.succeed(result_uri)
    except (Exception, SystemExit) as exc:
        run.fail(repr(exc))
        await _save(ctx, run)
        raise  # re-raise so arq logs it
    await _save(ctx, run)


class WorkerSettings:
    functions = [run_job]
    job_timeout = 1800
    # ponytail: 1800s covers CPU engines comfortably. Climb to a durable engine
    # (Temporal) only when a GPU training run genuinely needs multi-hour execution.
```

Also write `InlineEnqueuer`, which awaits the handler directly instead of pushing to Redis. Selected when `STUDIO_INLINE_JOBS=1`, so tests and local dev need no Valkey — the same dual-implementation trick as chem-cellar's `NullJobOrchestrator`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/integration/test_run_lifecycle.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/domain/execution backend/src/daikonstudio/infrastructure/worker.py backend/src/daikonstudio/application/execution backend/alembic/versions backend/tests
git commit -m "feat(execution): Run aggregate, arq worker and inline enqueuer"
```

---

### Task 14: Training orchestration

**Files:**
- Create: `backend/src/daikonstudio/application/execution/train_protocol.py`
- Test: `backend/tests/integration/test_train_protocol.py`

**Interfaces:**
- Consumes: `EngineRegistry` (Task 7), `Dataset` (Task 11), `InSilicoProtocol`/`derive_readouts` (Task 12), `Run` (Task 13), `BlobStore` (Task 5).
- Produces: `TrainProtocol(command, auth) -> Result[Run, DomainError]`, which creates one parent training Run and produces an `InSilicoProtocol` plus the raw material for a Scorecard.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/integration/test_train_protocol.py
def test_training_produces_a_draft_protocol_with_derived_readouts(studio, dataset):
    run = studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    studio.wait(run)
    protocol = studio.protocol_for(run)
    assert protocol.status.value == "draft"
    assert protocol.readouts[0].unit == dataset.target.unit


def test_training_always_also_trains_the_baseline(studio, dataset):
    """The baseline is mandatory, not a checkbox."""
    run = studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    studio.wait(run)
    scorecard = studio.scorecard_for(run)
    assert scorecard.baseline_metrics is not None
    assert scorecard.baseline_engine_id == "ecfp4-randomforest"


def test_a_scaffold_split_also_reports_the_random_split_number(studio, scaffold_dataset):
    """The optimism gap must be visible rather than inferred."""
    run = studio.train(dataset_id=scaffold_dataset.id, engine_id="ecfp4-xgboost", conditions={})
    studio.wait(run)
    scorecard = studio.scorecard_for(run)
    assert scorecard.random_split_metrics is not None


def test_a_random_split_reports_no_optimism_gap(studio, dataset):
    run = studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    studio.wait(run)
    assert studio.scorecard_for(run).random_split_metrics is None


def test_invalid_conditions_fail_the_run_with_a_useful_message(studio, dataset):
    run = studio.train(
        dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={"max_depth": 999}
    )
    studio.wait(run)
    assert studio.reload(run).status.value == "failed"
    assert "max_depth" in studio.reload(run).error_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_train_protocol.py -v`
Expected: FAIL — `TrainProtocol` not found.

- [ ] **Step 3: Write the orchestration**

The handler for `RunKind.TRAINING`, in order:

1. Guards, then load the Dataset and verify `require_same_workspace`.
2. `validate_conditions(manifest, command.conditions)` — a failure here fails the run with the message, before any compute.
3. Read the snapshot Parquet from the blob store into polars.
4. Derive the task from the Dataset, never from the values:
   `task = TaskType.BINARY_CLASSIFICATION if dataset.target.kind is TargetKind.BINARY else TaskType.REGRESSION`,
   and pass it on every `TrainContext` built below. Train the chosen engine via
   `asyncio.to_thread`. Report progress `0.33`, phase `"training <engine>"`.
5. Train the baseline engine on the identical frame and split. Progress `0.66`, phase `"training baseline"`.
6. If `dataset.split.strategy is SplitStrategy.SCAFFOLD`, re-assign a random split with the same seed and train the chosen engine once more for the optimism gap. Progress `0.9`.
7. Persist the chosen engine's artifact to `{workspace_id}/protocols/{protocol_id}/artifact/model.joblib`, create the `InSilicoProtocol` in DRAFT with `derive_readouts(dataset.target, task)`, and store the Scorecard.

```python
# ponytail: three fits per training request is free at ECFP4 speeds and will not be
# for GPU engines. When Phase 2 lands, make the random-split comparison opt-out for
# expensive engines. The baseline stays mandatory regardless.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/integration/test_train_protocol.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/execution/train_protocol.py backend/tests/integration/test_train_protocol.py
git commit -m "feat(execution): training orchestration with mandatory baseline and optimism gap"
```

---

### Task 15: Scorecard

**Files:**
- Create: `backend/src/daikonstudio/domain/execution/scorecard.py`
- Create: `backend/src/daikonstudio/application/execution/build_scorecard.py`
- Test: `backend/tests/unit/execution/test_scorecard.py`

**Interfaces:**
- Consumes: `nearest_neighbour_tanimoto`, `murcko_scaffold` (Task 6); `ValidationReport` (Task 9).
- Produces: `Scorecard(primary_metric, metrics, baseline_engine_id, baseline_metrics, random_split_metrics, noise_floor, worst_rows, applicability_coverage)`; `WorstRow(structure, actual, predicted, residual, scaffold)`; and:

```python
def build_scorecard(
    *,
    task: TaskType,
    actual: list[float],
    predicted: list[float],
    structures: list[str],
    train_structures: list[str],
    baseline_engine_id: str | None = None,
    baseline_metrics: dict[str, float] | None = None,
    random_split_metrics: dict[str, float] | None = None,
    duplicate_spread: float | None = None,
) -> Scorecard: ...
```

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/execution/test_scorecard.py
from daikonstudio.application.execution.build_scorecard import build_scorecard


from daikonstudio.application.engines.manifest import TaskType


def regression_card(**overrides):
    kwargs = {
        "task": TaskType.REGRESSION,
        "actual": [1.0, 2.0, 3.0],
        "predicted": [1.1, 2.1, 2.9],
        "structures": ["CCO", "CCN", "CCCO"],
        "train_structures": ["CCO"],
    }
    return build_scorecard(**{**kwargs, **overrides})


def test_regression_leads_with_rmse():
    card = regression_card()
    assert card.primary_metric == "rmse"
    assert set(card.metrics) >= {"rmse", "mae", "r2"}


def test_classification_leads_with_mcc_never_accuracy():
    """A 99.9%-negative dataset yields a 99.9%-accurate useless model."""
    card = build_scorecard(
        task=TaskType.BINARY_CLASSIFICATION,
        actual=[0.0] * 99 + [1.0],
        predicted=[0.0] * 100,
        structures=["CCO"] * 100,
        train_structures=["CCO"],
    )
    assert card.primary_metric == "mcc"
    assert set(card.metrics) >= {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert card.metrics["mcc"] == 0.0
    assert "accuracy" not in card.metrics


def test_worst_rows_are_ranked_by_residual_and_carry_their_scaffold():
    card = regression_card(
        actual=[1.0, 2.0, 9.0],
        predicted=[1.0, 2.0, 2.0],
        structures=["CCO", "CCN", "Cc1ccccc1"],
    )
    assert card.worst_rows[0].structure == "Cc1ccccc1"
    assert card.worst_rows[0].residual == 7.0
    assert card.worst_rows[0].scaffold == "c1ccccc1"


def test_applicability_coverage_is_the_fraction_above_the_threshold():
    card = regression_card(
        actual=[1.0, 2.0],
        predicted=[1.0, 2.0],
        structures=["CCO", "CCCCCCCCCCCCCCCC"],
        train_structures=["CCO"],
    )
    assert card.applicability_coverage == 0.5


def test_baseline_and_optimism_gap_are_carried_through_when_supplied():
    card = regression_card(
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"rmse": 0.9},
        random_split_metrics={"rmse": 0.2},
    )
    assert card.baseline_engine_id == "ecfp4-randomforest"
    assert card.random_split_metrics["rmse"] == 0.2


def test_noise_floor_is_absent_when_there_were_no_duplicates():
    assert regression_card(duplicate_spread=None).noise_floor is None
    assert regression_card(duplicate_spread=0.4).noise_floor == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `build_scorecard`**

Regression metrics: RMSE (primary), MAE, R². Classification metrics: MCC (primary), balanced accuracy, AUROC, AUPRC. **Never bare accuracy** — it is not computed at all, so it cannot be reported by mistake.

`worst_rows` is the twenty largest absolute residuals, each carrying its structure and Murcko scaffold so the frontend can group them and a chemist reads "it fails on the sulfonamides".

`applicability_coverage` is the fraction of test structures whose `nearest_neighbour_tanimoto` to the training set is at least 0.3, and each row also carries its own similarity so the triage grid can flag individual compounds.

`noise_floor` is passed through from the Dataset's `duplicate_spread` and is `None` for binary targets, which have no equivalent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/execution -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/domain/execution/scorecard.py backend/src/daikonstudio/application/execution/build_scorecard.py backend/tests/unit/execution
git commit -m "feat(execution): Scorecard with baseline comparison, optimism gap and applicability"
```

---

### Task 16: Protocol routes and publishing

**Files:**
- Create: `backend/src/daikonstudio/interface/routes/protocols.py`
- Create: `backend/src/daikonstudio/application/catalog/{publish_protocol,get_scorecard,list_protocols}.py`
- Test: `backend/tests/api/test_protocols.py`

**Interfaces:**
- Consumes: `TrainProtocol` (Task 14), `Scorecard` (Task 15), `InSilicoProtocol` (Task 12).
- Produces: `POST /api/v1/protocols` (202 → Run), `GET /api/v1/protocols`, `GET /api/v1/protocols/{id}`, `GET /api/v1/protocols/{id}/scorecard`, `POST /api/v1/protocols/{id}/publish` (204).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_protocols.py
def test_training_request_returns_202_with_a_run(client, dataset_id):
    response = client.post("/api/v1/protocols", json={
        "name": "solubility rf", "dataset_id": dataset_id,
        "engine_id": "ecfp4-randomforest", "conditions": {},
    })
    assert response.status_code == 202
    assert response.json()["status"] == "pending"


def test_publish_locks_the_protocol(client, trained_protocol_id):
    assert client.post(f"/api/v1/protocols/{trained_protocol_id}/publish").status_code == 204
    assert client.get(f"/api/v1/protocols/{trained_protocol_id}").json()["is_locked"] is True


def test_publishing_twice_returns_423_locked(client, trained_protocol_id):
    client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    assert client.post(f"/api/v1/protocols/{trained_protocol_id}/publish").status_code == 423


def test_scorecard_exposes_the_baseline_comparison(client, trained_protocol_id):
    card = client.get(f"/api/v1/protocols/{trained_protocol_id}/scorecard").json()
    assert card["baseline_engine_id"] == "ecfp4-randomforest"
    assert card["primary_metric"] in {"rmse", "mcc"}
    assert len(card["worst_rows"]) <= 20


def test_publish_requires_the_editor_role(viewer_client, trained_protocol_id):
    assert viewer_client.post(f"/api/v1/protocols/{trained_protocol_id}/publish").status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_protocols.py -v`
Expected: FAIL — 404 on POST.

- [ ] **Step 3: Write the routes**

`APIRouter(prefix="/api/v1/protocols", tags=["protocols"])`. Each handler calls one use case and passes the result through `result_to_response`. `DataLockedError` maps to 423 via the shared error map from Task 2, so republishing needs no special-casing in the route.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_protocols.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/interface/routes/protocols.py backend/src/daikonstudio/application/catalog backend/tests/api/test_protocols.py
git commit -m "feat(catalog): protocol routes, publishing and scorecard endpoint"
```

---

### Task 17: Prediction runs with content-addressed caching

**Files:**
- Create: `backend/src/daikonstudio/application/execution/predict_with_protocol.py`
- Create: `backend/src/daikonstudio/interface/routes/runs.py`
- Test: `backend/tests/integration/test_prediction_cache.py`, `backend/tests/api/test_runs.py`

**Interfaces:**
- Consumes: `InSilicoProtocol` (Task 12), `Run`/`compute_cache_key` (Task 13), `EngineRegistry` (Task 7).
- Produces: `PredictWithProtocol(command, auth) -> Result[Run, DomainError]`; `POST /api/v1/runs` (202), `GET /api/v1/runs/{id}`, `GET /api/v1/runs/{id}/results`, `POST /api/v1/runs/{id}/cancel` (204).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/integration/test_prediction_cache.py
def test_identical_inputs_reuse_the_cached_run(studio, published_protocol, upload_ref):
    first = studio.predict(published_protocol.id, upload_ref)
    studio.wait(first)
    second = studio.predict(published_protocol.id, upload_ref)
    assert second.result_uri == studio.reload(first).result_uri
    assert second.status.value == "ready"  # never re-queued


def test_different_conditions_produce_a_different_cache_key(studio, published_protocol, upload_ref):
    first = studio.predict(published_protocol.id, upload_ref, conditions={"threshold": 0.5})
    second = studio.predict(published_protocol.id, upload_ref, conditions={"threshold": 0.7})
    assert first.cache_key != second.cache_key


def test_a_new_protocol_version_invalidates_the_cache(studio, published_protocol, upload_ref):
    first = studio.predict(published_protocol.id, upload_ref)
    v2 = studio.publish_new_version(published_protocol)
    second = studio.predict(v2.id, upload_ref)
    assert first.cache_key != second.cache_key


def test_running_an_unpublished_protocol_is_rejected(studio, draft_protocol, upload_ref):
    result = studio.predict_raw(draft_protocol.id, upload_ref)
    assert result.failure().__class__.__name__ == "ConflictError"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_prediction_cache.py -v`
Expected: FAIL — `PredictWithProtocol` not found.

- [ ] **Step 3: Write the use case**

```python
def compute_cache_key(*, protocol_id, protocol_version, input_hash, conditions) -> str:
    payload = json.dumps(
        {"p": str(protocol_id), "v": protocol_version, "i": input_hash,
         "c": sorted(conditions.items())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
```

Before enqueuing, look up a `READY` Run in the workspace with the same `cache_key`; on a hit, return it directly. Only a *published* protocol may be run — a draft raises `ConflictError`. This is chem-cellar's `UmapJob` pattern with the model version folded into the key.

- [ ] **Step 4: Write the routes**

`GET /api/v1/runs/{id}/results` returns `PaginatedResponse[PredictionResponse]` read from the results Parquet, each row carrying `structure`, one field per Readout, `uncertainty` and `applicability` so the triage grid can render and filter without a second call.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/integration/test_prediction_cache.py tests/api/test_runs.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/application/execution/predict_with_protocol.py backend/src/daikonstudio/interface/routes/runs.py backend/tests
git commit -m "feat(execution): prediction runs with content-addressed result caching"
```

---

### Task 18: Engines route

**Files:**
- Create: `backend/src/daikonstudio/interface/routes/engines.py`
- Test: `backend/tests/api/test_engines.py`

**Interfaces:**
- Consumes: `EngineRegistry` (Task 7).
- Produces: `GET /api/v1/engines` returning the manifest list, including each engine's `conditions` so the frontend renders the form from the response.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_engines.py
def test_engines_are_listed_with_their_conditions(client):
    engines = client.get("/api/v1/engines").json()
    ids = {e["id"] for e in engines}
    assert ids == {"ecfp4-xgboost", "ecfp4-randomforest"}
    xgb = next(e for e in engines if e["id"] == "ecfp4-xgboost")
    keys = {c["key"] for c in xgb["conditions"]}
    assert keys == {"n_estimators", "max_depth", "learning_rate"}
    depth = next(c for c in xgb["conditions"] if c["key"] == "max_depth")
    assert depth["type"] == "integer" and depth["default"] == 6


def test_exactly_one_engine_is_marked_as_the_baseline(client):
    engines = client.get("/api/v1/engines").json()
    assert sum(1 for e in engines if e["is_baseline"]) == 1
```

The backend describes its own engine catalog and the frontend renders the picker and the condition form from the response — chem-cellar's `GET /search/algorithms` pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_engines.py -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write the route**

```python
router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


@router.get("", response_model=list[EngineManifestResponse])
async def list_engines(registry: EngineRegistry = Depends(get_engine_registry)):
    return [EngineManifestResponse.from_manifest(m) for m in registry.manifests()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_engines.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/interface/routes/engines.py backend/tests/api/test_engines.py
git commit -m "feat(catalog): self-describing engines endpoint"
```

---

### Task 19: Collections and export

**Files:**
- Create: `backend/src/daikonstudio/domain/data/collection.py`
- Create: `backend/src/daikonstudio/application/data/{create_collection,export_collection}.py`
- Create: `backend/src/daikonstudio/interface/routes/collections.py`
- Create: `backend/alembic/versions/005_collections.py`
- Test: `backend/tests/api/test_collections.py`

**Interfaces:**
- Consumes: `Run` (Task 13), `BlobStore` (Task 5).
- Produces: `Collection(AggregateRoot)` with `name`, `derived_from_run_id`, `member_count`, `snapshot_uri`; `POST /api/v1/collections` (201), `GET /api/v1/collections/{id}/export?format=csv|sdf`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/api/test_collections.py
def test_saving_a_triage_selection_creates_a_collection(client, ready_run_id):
    response = client.post("/api/v1/collections", json={
        "name": "top 2 for synthesis", "run_id": ready_run_id, "row_ids": [0, 3],
    })
    assert response.status_code == 201
    assert response.json()["member_count"] == 2
    assert response.json()["derived_from_run_id"] == ready_run_id


def test_csv_export_contains_the_selected_structures(client, collection_id):
    response = client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "smiles" in response.text.splitlines()[0]


def test_sdf_export_is_a_valid_molfile_block(client, collection_id):
    response = client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    assert response.text.count("$$$$") == 2


def test_predictions_carry_the_ai_predicted_provenance(client, collection_id):
    body = client.get(f"/api/v1/collections/{collection_id}").json()
    assert body["provenance"]["generation_method"] == "ai_predicted"


def test_selecting_rows_from_an_unfinished_run_is_rejected(client, pending_run_id):
    response = client.post("/api/v1/collections", json={
        "name": "too early", "run_id": pending_run_id, "row_ids": [0],
    })
    assert response.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/api/test_collections.py -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write the aggregate and use cases**

`CreateCollection` reads the run's results Parquet, filters to `row_ids`, writes a Collection snapshot, and stamps `Provenance(generation_method=GenerationMethod.AI_PREDICTED)` carrying the protocol id and version. A run not in `READY` raises `ConflictError` → 409.

`ExportCollection` renders CSV via polars, and SDF via `rdkit.Chem.SDWriter` with each readout written as an SD tag named after the readout, including its unit — so a predicted IC50 arrives in the downstream tool labelled the same way a measured one is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/api/test_collections.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/domain/data/collection.py backend/src/daikonstudio/application/data backend/src/daikonstudio/interface/routes/collections.py backend/alembic/versions backend/tests/api/test_collections.py
git commit -m "feat(data): Collections from triage selections with CSV and SDF export"
```

---

### Task 20: End-to-end acceptance test

**Files:**
- Create: `backend/tests/integration/test_full_loop.py`
- Create: `backend/tests/fixtures/pains_sample.csv`
- Modify: `backend/README.md`

**Interfaces:**
- Consumes: everything.
- Produces: the test that defines "Phase 1 backend is done".

- [ ] **Step 1: Build the fixture**

Take 200 rows from `/Users/sidx/workspace/lab-ai/pains/dataset/balanced_training_set.csv` — a real dataset with real problems — and commit them as `backend/tests/fixtures/pains_sample.csv` with a `smiles` column and a binary `is_pains` column.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/integration/test_full_loop.py
def test_a_scientist_can_walk_the_whole_loop(client, csv_upload):
    """Spec §10 criterion 4: every noun exercised end to end."""
    upload_ref = csv_upload(open("tests/fixtures/pains_sample.csv", "rb").read())

    dataset = client.post("/api/v1/datasets", json={
        "name": "PAINS", "upload_ref": upload_ref, "structure_column": "smiles",
        "target": {"column": "is_pains", "kind": "binary"},
        "split": {"strategy": "scaffold", "seed": 42},
    }).json()
    assert dataset["row_count"] > 0

    run = client.post("/api/v1/protocols", json={
        "name": "PAINS xgb", "dataset_id": dataset["id"],
        "engine_id": "ecfp4-xgboost", "conditions": {},
    }).json()
    protocol_id = _poll_until_ready(client, run["id"])["protocol_id"]

    card = client.get(f"/api/v1/protocols/{protocol_id}/scorecard").json()
    assert card["primary_metric"] == "mcc"
    assert card["baseline_metrics"]["mcc"] is not None
    assert card["random_split_metrics"] is not None  # scaffold split -> optimism gap

    assert client.post(f"/api/v1/protocols/{protocol_id}/publish").status_code == 204

    predict_ref = csv_upload(b"smiles\nCCO\nc1ccccc1\nCc1ccccc1\n")
    prediction = client.post("/api/v1/runs", json={
        "protocol_id": protocol_id, "upload_ref": predict_ref,
    }).json()
    _poll_until_ready(client, prediction["id"])

    results = client.get(f"/api/v1/runs/{prediction['id']}/results").json()
    assert len(results["items"]) == 3
    assert "applicability" in results["items"][0]

    collection = client.post("/api/v1/collections", json={
        "name": "flagged", "run_id": prediction["id"], "row_ids": [0, 1],
    }).json()
    export = client.get(f"/api/v1/collections/{collection['id']}/export?format=csv")
    assert export.status_code == 200 and len(export.text.splitlines()) == 3
```

- [ ] **Step 3: Run the test**

Run: `cd backend && uv run pytest tests/integration/test_full_loop.py -v`
Expected: PASS. Fix whatever it exposes — this test is the acceptance gate, and it asserts on the Scorecard's *structure* rather than on absolute metric values, so it does not become flaky when a library version changes.

- [ ] **Step 4: Run the whole suite and every check**

Run: `cd backend && uv run pytest -v && uv run ruff check . && uv run mypy src && uv run lint-imports`
Expected: all pass, all four import-linter contracts KEPT.

- [ ] **Step 5: Write the README**

Document `make install`, `make up`, `make dev-be`, `make dev-worker`, `make migrate`, `make test`, the port table (8002/5435/6381), and the `STUDIO_INLINE_JOBS=1` escape hatch for running without Valkey.

- [ ] **Step 6: Generate the OpenAPI snapshot for the UI plan**

Run:

```bash
cd backend && uv run python -c "
import json
from daikonstudio.interface.app import create_app
print(json.dumps(create_app().openapi(), indent=2))
" > ../frontend-openapi.json
```

This snapshot is the input to the frontend plan's orval config — committed, reviewable in diffs, and it works offline and in CI.

- [ ] **Step 7: Commit**

```bash
git add backend/tests backend/README.md frontend-openapi.json
git commit -m "test: end-to-end acceptance for the Phase 1 curation loop"
```

---

## Self-Review

**Spec coverage.** Every §3 "In" item maps to a task: CSV source → 11; validation, dedup, split, snapshot → 9, 10, 11; two engines → 8; training with derived readouts → 12, 14; Scorecard with baseline, optimism gap, noise floor, worst-20, applicability → 15; publish → 12, 16; prediction runs → 17; triage grid data and Collections and export → 17, 19; Sentinel auth and workspace tenancy → 4; lineage captured → 11, 13, 14, 19 (`derived_from_run_id`, `dataset_id`, `protocol_id` on every produced entity). §6's API surface is covered by tasks 11, 16, 17, 18, 19. §8's two studio-specific error cases are covered in 11 (422 carrying the report) and 13 (failed run re-raises for arq).

**Deferred deliberately, matching spec §3 "Out".** No GPU, containers, chem-cellar source, Proposal, generation, ensembles, Sweep UI, or third-party engine registration appears in any task.

**Known gap, accepted.** `Sweep` is defined in the spec's terminology but has no aggregate here; the mandatory-baseline mechanism in Task 14 is a two-run sweep in all but name. This matches the spec's own open-questions note and is scheduled for Phase 2.

**Type consistency.** `TaskType`, `ConditionType`, `ConditionSpec`, `EngineManifest`, `TrainContext`, `TrainResult`, `PredictContext` are defined in Task 7 and used unchanged in 8, 12, 14, 17, 18. `TargetSpec`/`TargetKind`/`Direction` defined in Task 9, used in 10, 11, 12, 15. `SplitSpec`/`SplitStrategy` defined in Task 10, used in 11, 14. `RunKind`/`RunStatus`/`Run` defined in Task 13, used in 14, 17, 19. `BlobStore` defined in Task 5, used in 10, 11, 14, 17, 19. The engine ids `ecfp4-xgboost` and `ecfp4-randomforest` are spelled identically in tasks 8, 14, 16, 18 and 20.

## Follow-on plan

The frontend plan (`2026-07-25-phase-1-frontend.md`) is deliberately **not** written yet. It should be authored after Task 20 produces `frontend-openapi.json`, so orval generates against the real API surface rather than a guessed one. It will cover: Next.js 16 scaffold with `@structflo/daikon-design-tokens` and the runtime `/api/config` route, Sentinel JS SDK wiring and the `/api/auth/mint` BFF, the engine catalog, the dataset upload wizard with its validation report, the training form rendered from engine manifests, the Scorecard page, the AG Grid triage view with RDKit WASM structure rendering, and Collection export.
