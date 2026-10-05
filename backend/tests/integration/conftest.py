"""Reuses the real-auth HTTP harness from `tests/api/conftest.py` rather than
duplicating or -- worse -- lightening it for this directory: Task 20's brief
is explicit that the acceptance test must not bypass authentication.
Importing the same fixture functions (not reimplementing a parallel harness)
is what guarantees `test_full_loop.py` exercises the identical
`AuthzMiddleware` path the API test suite does, with zero chance of drift
between "real" auth and "acceptance-test" auth.

pytest resolves fixtures by scanning each conftest module's namespace for
the `@pytest.fixture`-marked callables it holds, regardless of where they
were originally defined -- a plain import re-exports them here exactly as
if they were declared in this file.
"""

from tests.api.conftest import (  # noqa: F401
    _resolve_jwks_locally,
    admin_client,
    anonymous_client,
    app,
    auth_headers,
    client,
    client_user_id,
    csv_upload,
    protocol_access,
    session_factory,
    signing_key,
    workspace_id,
)
