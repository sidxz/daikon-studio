"""One place that decides what a domain failure looks like over HTTP.

Both halves of the railway land here: a `Failure` is re-raised by
`result_to_response` and an error raised by a guard propagates on its own, so a
route never has to know which kind it is holding.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from returns.result import Failure, Result, Success

from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    ConcurrencyConflictError,
    ConflictError,
    DataLockedError,
    DomainError,
    GoneError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)

# Order matters: the first match wins, so subclasses must precede their bases.
_ERROR_STATUS_MAP: list[tuple[type[DomainError], int]] = [
    (NotFoundError, 404),
    (ValidationError, 422),
    (ConcurrencyConflictError, 409),
    (ConflictError, 409),
    (AuthorizationError, 403),
    (DataLockedError, 423),
    (GoneError, 410),
    (ServiceUnavailableError, 503),
]


def _error_to_status(error: DomainError) -> int:
    for error_type, status in _ERROR_STATUS_MAP:
        if isinstance(error, error_type):
            return status
    return 500


def _error_to_body(error: DomainError) -> dict[str, Any]:
    body = error.to_body()
    if isinstance(error, ConcurrencyConflictError):
        body["retry"] = True
    return body


def register_error_handlers(app: FastAPI, *, cors_origins: list[str] | None = None) -> None:
    allowed_origins = set(cors_origins or [])
    logger = structlog.get_logger(__name__)

    @app.exception_handler(DomainError)
    async def _domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        headers = {"Retry-After": "1"} if isinstance(exc, ConcurrencyConflictError) else {}
        return JSONResponse(
            status_code=_error_to_status(exc), content=_error_to_body(exc), headers=headers
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """A bug is a JSON 500 that names its request id, never a stack trace.

        Starlette's ServerErrorMiddleware sits *outside* CORSMiddleware, so a
        500 would otherwise reach the browser with no CORS headers and read as
        a network error instead of a server error. The allowed origin is
        echoed here by hand for exactly that reason.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error", request_id=request_id, path=request.url.path)
        # RequestIdMiddleware never regains control once the exception has
        # propagated past it, so the id is stamped here too.
        headers: dict[str, str] = {"X-Request-ID": request_id} if request_id else {}
        origin = request.headers.get("origin")
        if origin is not None and origin in allowed_origins:
            headers |= {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Vary": "Origin",
            }
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": (
                    "An unexpected server error occurred. Include the request ID if you report it."
                ),
                "request_id": request_id,
            },
            headers=headers,
        )


def result_to_response[T](result: Result[T, DomainError]) -> T:
    """Unwrap a Success, or raise the Failure for the handler above to render."""
    match result:
        case Success(value):
            return value  # type: ignore[no-any-return]
        case Failure(error):
            raise error
        case _:  # pragma: no cover -- Result has exactly two shapes
            raise RuntimeError(f"Unexpected Result: {result!r}")
