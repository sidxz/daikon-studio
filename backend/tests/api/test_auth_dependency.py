import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.interface.dependencies._core import AuthDep


def test_get_auth_rejects_when_sentinel_not_configured():
    """No STUDIO_SENTINEL_SERVICE_KEY in the test env: the wired-in dependency
    must be the reject-all stub, never a silent bypass for an unauthenticated caller.
    """
    app = FastAPI()

    @app.get("/_probe")
    async def probe(auth: AuthDep) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app, raise_server_exceptions=True)
    with pytest.raises(ServiceUnavailableError):
        client.get("/_probe")
