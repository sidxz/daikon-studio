"""The startup warning that makes a silent auth wedge visible.

Regression test for a real incident (2026-08-07): every authenticated request
returned 403 "Authz token was issued for a different service" with a completely
clean backend log. The SDK's `fetch_whoami()` swallows every failure and returns
None, leaving `effective_scope` as the service's own name, and it never retries --
so one transient blip during any of an editing session's `--reload` cycles wedges
the process until it is restarted, with no evidence anywhere.

These tests are about the *log line*, not about Duar: they assert that the
ambiguous state announces itself.
"""

from __future__ import annotations

import logging

from daikonstudio.infrastructure.duar.auth import log_effective_scope


class _FakeDuar:
    """Only the two attributes `log_effective_scope` reads.

    A stub rather than a real Duar: building one needs a service key and
    reaches a remote host, and neither is what this behaviour depends on.
    """

    def __init__(self, *, service_name: str, effective_scope: str) -> None:
        self.service_name = service_name
        self.effective_scope = effective_scope


def test_warns_when_the_scope_never_moved_off_the_service_name(caplog) -> None:
    """The wedged state: whoami failed (or found no realm), so the process will
    reject every token minted for the realm. This is the line whose absence cost a
    long investigation."""
    duar = _FakeDuar(service_name="daikon-studio-dev", effective_scope="daikon-studio-dev")

    with caplog.at_level(logging.WARNING):
        log_effective_scope(duar)  # type: ignore[arg-type]

    assert any(record.levelno == logging.WARNING for record in caplog.records)
    message = caplog.text
    assert "daikon-studio-dev" in message
    # The operator needs to be told the cure, not just the symptom.
    assert "restart" in message.lower()
    assert "403" in message


def test_does_not_warn_once_the_realm_scope_is_resolved(caplog) -> None:
    """The healthy state: whoami re-pointed the scope at the shared realm slug, so
    tokens carrying `svc: daikon-siblings` are accepted. No warning."""
    duar = _FakeDuar(service_name="daikon-studio-dev", effective_scope="daikon-siblings")

    with caplog.at_level(logging.WARNING):
        log_effective_scope(duar)  # type: ignore[arg-type]

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []
