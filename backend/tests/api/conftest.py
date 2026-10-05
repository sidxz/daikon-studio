"""API-test harness: a real app, real auth, a real database, a temp blob store.

Auth is NOT stubbed. ``AuthzMiddleware`` runs exactly as in production and still
validates two RS256-signed JWTs on every request; the only thing replaced is
*where the signing key comes from* (``PyJWKClient.get_signing_key_from_jwt``),
because the middleware fetches JWKS over sync urllib and there is no identity
service listening in tests. A request with no tokens, expired tokens, or tokens
signed by another key is rejected by the real middleware — see
``test_unauthenticated_request_is_rejected``.

The app is driven through ``httpx.ASGITransport`` rather than ``TestClient``:
the app then runs on the *test's* event loop, so the asyncpg connection the
database fixture opens is usable from inside a request handler. ``TestClient``
would run the app on its own portal thread/loop and the connection would belong
to the wrong loop. A side effect worth naming: ASGITransport does not run the
lifespan, so the lifespan's ``fetch_duar_public_key()`` — which would hang
against a dead ``duar_url`` — never fires. Nothing in the request path needs
it: ``AuthzMiddleware`` resolves keys per request, and the DI container is built
in ``create_app()``, not in the lifespan.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKClient
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from tests.fakes.protocol_access import FakeProtocolAccess

from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings

_AUTHZ_AUDIENCE = "duar:authz"


@pytest.fixture(scope="session")
def signing_key() -> tuple[str, str]:
    """One RSA keypair for the whole run — generating it per test is slow."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private_pem, public_pem


@pytest.fixture(autouse=True)
def _resolve_jwks_locally(monkeypatch, signing_key) -> None:
    """Point both PyJWKClients (IdP and Duar) at the test keypair.

    Only key *lookup* is replaced. Signature, audience, issuer, expiry, the
    idp_sub binding and the svc binding are all still enforced by the real
    middleware against the real token.
    """
    _, public_pem = signing_key

    class _Key:
        key = public_pem

    monkeypatch.setattr(PyJWKClient, "get_signing_key_from_jwt", lambda self, token: _Key())


def auth_headers(
    private_pem: str,
    *,
    workspace_id: uuid.UUID,
    role: str = "editor",
    user_id: uuid.UUID | None = None,
) -> dict[str, str]:
    """Mint the IdP + Duar authz token pair AuthzMiddleware expects."""
    settings = Settings()
    now = dt.datetime.now(dt.UTC)
    expires = now + dt.timedelta(minutes=5)
    idp_sub = f"test-idp-sub-{uuid.uuid4()}"
    idp_token = jwt.encode(
        {
            "sub": idp_sub,
            "email": "scientist@example.org",
            "name": "Test Scientist",
            "aud": settings.idp_audience,
            "iss": settings.idp_issuer,
            "iat": now,
            "exp": expires,
        },
        private_pem,
        algorithm="RS256",
    )
    authz_token = jwt.encode(
        {
            "sub": str(user_id or uuid.uuid4()),
            "idp_sub": idp_sub,
            "wid": str(workspace_id),
            "wslug": "test-workspace",
            "wrole": role,
            # Must resolve the same way build_duar() does. Using `service_name`
            # alone mints a token for the display name, so any deployment that
            # registers under a different Duar identity (a `-dev` instance in the
            # shared realm, say) gets 403 "Authz token was issued for a different
            # service" on every request -- the harness would be testing the wrong app.
            "svc": settings.duar_service_name or settings.service_name,
            "aud": _AUTHZ_AUDIENCE,
            "iat": now,
            "exp": expires,
        },
        private_pem,
        algorithm="RS256",
    )
    return {"Authorization": f"Bearer {idp_token}", "X-Authz-Token": authz_token}


@pytest.fixture
def workspace_id() -> uuid.UUID:
    """A fresh workspace per test — no test can see another test's rows."""
    return uuid.uuid4()


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """One connection and one outer transaction per test, rolled back at teardown.

    Same recipe as the ``migrated_session`` fixture in the root conftest, one level
    up: the app needs a *factory* because its repository opens a session per call,
    and every one of those must join the transaction this fixture rolls back.
    ``join_transaction_mode="create_savepoint"`` makes the repository's own
    ``commit()`` land on a savepoint, so the rollback below still discards it.
    """
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


@pytest.fixture
def protocol_access() -> FakeProtocolAccess:
    return FakeProtocolAccess()


@pytest.fixture
def app(tmp_path, session_factory, protocol_access):
    """The real app, with the database and blob store pointed at test-owned ones.

    Overriding is a child container: lagom refuses a second ``define`` on the same
    container, so a test cannot silently reassign production wiring in place.
    """
    application = create_app()
    # inline_jobs=True: training runs in-process via InlineEnqueuer rather than
    # waiting for a self-hosted runner to claim it, so `POST /api/v1/protocols`
    # needs no runner in tests (see infrastructure/jobs.py's module docstring
    # for both implementations).
    container = Container(
        create_container(Settings(blob_base_url=f"file://{tmp_path}", inline_jobs=True))
    )
    container.define(async_sessionmaker, Singleton(lambda: session_factory))
    container.define(ProtocolAccess, Singleton(lambda: protocol_access))  # type: ignore[type-abstract]
    application.state.container = container
    return application


def _client(app, headers: dict[str, str] | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers=headers or {},
    )


@pytest_asyncio.fixture
async def client(app, signing_key, workspace_id) -> AsyncIterator[httpx.AsyncClient]:
    headers = auth_headers(signing_key[0], workspace_id=workspace_id)
    async with _client(app, headers) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def other_workspace_client(app, signing_key) -> AsyncIterator[httpx.AsyncClient]:
    """A second tenant, same app, same database — the tenancy-isolation probe."""
    headers = auth_headers(signing_key[0], workspace_id=uuid.uuid4())
    async with _client(app, headers) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def viewer_client(app, signing_key, workspace_id) -> AsyncIterator[httpx.AsyncClient]:
    headers = auth_headers(signing_key[0], workspace_id=workspace_id, role="viewer")
    async with _client(app, headers) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def admin_client(app, signing_key, workspace_id) -> AsyncIterator[httpx.AsyncClient]:
    """Runner management is an admin action; everything else uses `client` (editor)."""
    headers = auth_headers(signing_key[0], workspace_id=workspace_id, role="admin")
    async with _client(app, headers) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def other_editor_client(app, signing_key, workspace_id) -> AsyncIterator[httpx.AsyncClient]:
    """A second editor in the same workspace: may read everything, delete only their own."""
    headers = auth_headers(signing_key[0], workspace_id=workspace_id, role="editor")
    async with _client(app, headers) as http_client:
        yield http_client


@pytest_asyncio.fixture
async def anonymous_client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with _client(app, None) as http_client:
        yield http_client


@pytest.fixture
def csv_upload(client) -> Iterator[Callable[[bytes], object]]:
    """Upload raw CSV bytes and hand back the opaque ``upload_ref``."""

    async def _upload(data: bytes) -> str:
        response = await client.post(
            "/api/v1/datasets/uploads",
            files={"file": ("data.csv", data, "text/csv")},
        )
        assert response.status_code == 201, response.text
        return str(response.json()["upload_ref"])

    yield _upload
