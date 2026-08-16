import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.infrastructure.duar import auth as duar_auth
from daikonstudio.interface.dependencies import _core


async def test_duar_not_configured_stub_rejects_never_bypasses():
    """The stub raises on its own. Necessary but not sufficient: this alone
    would still pass even if the *selection* logic below (which decides
    whether this stub gets wired in at all) were broken. See the end-to-end
    test below for that.
    """
    with pytest.raises(ServiceUnavailableError):
        await _core._duar_not_configured()


def test_auth_dep_rejects_end_to_end_when_duar_unconfigured(monkeypatch):
    """Force _core.py's real module-level selection (the try/except around
    get_duar()) to hit the unconfigured branch, then prove a route wired
    with the public AuthDep actually rejects through FastAPI's real dependency
    resolution — not just that the stub function raises in isolation.

    This is the check that fails if someone widens the `except` clause in
    _core.py and falls through to a bypass instead of the reject-all stub.
    """

    def _raise_unconfigured() -> object:
        raise ValueError("service_key is required")  # matches Duar.__init__'s own message

    monkeypatch.setattr(duar_auth, "get_duar", _raise_unconfigured)
    importlib.reload(_core)  # re-run the try/except under the patched get_duar
    try:
        app = FastAPI()

        @app.get("/_probe")
        async def probe(auth: _core.AuthDep) -> dict[str, str]:
            return {"ok": "true"}

        client = TestClient(app, raise_server_exceptions=True)
        with pytest.raises(ServiceUnavailableError):
            client.get("/_probe")
    finally:
        monkeypatch.undo()
        importlib.reload(_core)  # restore the real, configured module for later tests
