import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.interface.app import check_database, create_app
from daikonstudio.settings import Settings


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


async def test_every_response_carries_a_request_id(client):
    response = await client.get("/health")
    assert response.headers["x-request-id"]


async def test_a_supplied_request_id_is_echoed(client):
    response = await client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


async def test_ready_is_200_when_the_database_answers(client):
    response = await client.get("/ready")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ready"}


async def test_check_database_names_the_failure_without_raising():
    unbound = async_sessionmaker(bind=None)  # no engine behind it: executing raises
    assert await check_database(unbound) is not None


async def test_an_unhandled_error_is_a_json_500_with_cors_and_request_id(app, client):
    """A bug must reach the browser as a 500 it can read -- with CORS headers, or
    the UI shows a network error -- and with the request id the log line carries."""

    @app.get("/__boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    origin = Settings().cors_origins[0]
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=dict(client.headers)
    ) as lenient:
        response = await lenient.get("/__boom", headers={"Origin": origin, "X-Request-ID": "r-1"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalError"
    assert body["request_id"] == "r-1"
    assert "kaboom" not in response.text
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["x-request-id"] == "r-1"
