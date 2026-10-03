"""What a workspace member reads on a failed run.

`repr(exc)` used to be stored verbatim and shown to every viewer of the run,
which carried blob paths and internal URLs to the browser. Domain errors are
written for users and pass through. A `ValueError`/`TypeError` from a library
names the data problem (`Input y contains NaN.` is actionable) and keeps its
text, without the class name. Everything else is reduced to its class name; the traceback is in the
server log, under the run id.
"""

from __future__ import annotations

from daikonstudio.domain.shared.errors import DomainError

_MAX_LENGTH = 500


def user_facing_error(exc: BaseException) -> str:
    if isinstance(exc, DomainError):
        return f"{exc.message} ({exc.detail})" if exc.detail else exc.message
    if isinstance(exc, ValueError | TypeError) and str(exc):
        return str(exc)[:_MAX_LENGTH]
    return (
        f"The run failed with an unexpected internal error ({type(exc).__name__}). "
        "Details are in the runner log."
    )
