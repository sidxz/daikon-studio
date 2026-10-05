"""Who may see a Run. A prediction runs a published protocol, so the workspace may see
it. A training run shows a draft's name and metrics, so it follows its protocol's
privacy."""

from __future__ import annotations

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.domain.execution.run import Run, RunKind


async def run_visible(run: Run, auth: AuthContext | None, access: ProtocolAccess) -> bool:
    """A prediction runs a published protocol, so the workspace may see it. A training
    run shows a draft's name and metrics, so it is visible to whoever started it, and
    otherwise to anyone who may view the protocol it produced. While it has none, only
    those with full access (admins, owners) see it. `auth=None` is a system call."""
    if auth is None or run.kind is RunKind.PREDICTION or run.requested_by == auth.user_id:
        return True
    if run.protocol_id is None:
        return await access.visible_ids(auth) is None
    return await access.can_view(auth, run.protocol_id)
