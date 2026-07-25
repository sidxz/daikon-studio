import pytest

from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.interface.dependencies._core import _sentinel_not_configured


async def test_sentinel_not_configured_stub_rejects_never_bypasses():
    """The dependency wired in when Sentinel is unconfigured/misconfigured (see
    _core.py) must reject every request — never let one through unauthenticated.
    """
    with pytest.raises(ServiceUnavailableError):
        await _sentinel_not_configured()
