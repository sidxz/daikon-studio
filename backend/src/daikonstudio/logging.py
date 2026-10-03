"""The one place logging is configured, for the API process and the runner agent.

Nothing in this app configured logging before 2026-10-02: the root logger had no
handler, so Python's `lastResort` fallback emitted WARNING and above to stderr and
dropped INFO -- which is why the Duar realm-scope line never appeared and the auth
wedge of 2026-08-07 took a long investigation. stdlib and structlog records both
render through one structlog formatter: console text for a terminal, JSON for a
log collector.
"""

from __future__ import annotations

import logging
import re
import sys

import structlog

#: Paths whose *successful* access lines say nothing: two runners polling every
#: 3 s is 40 lines a minute. A failing poll or probe still logs.
NOISY_PATHS = ("/api/v1/runner/claim", "/health", "/ready")

# Uvicorn's access line is `<client> - "<method> <path> HTTP/<v>" <status>`, with
# the status code last and nothing after it -- so the match must accept end-of-line.
_SUCCESS = re.compile(r'" (200|204)(\s|$)')


class DropNoisyAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        quiet_path = any(f" {path} " in message for path in NOISY_PATHS)
        return not (quiet_path and _SUCCESS.search(message) is not None)


def configure_logging(*, level: str = "INFO", fmt: str = "console") -> None:
    """Idempotent: safe to call from `create_app()` on every `--reload`."""
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Uvicorn installs its own handlers before the app imports; route them
    # through ours so every line has one shape, and mute the poll chatter.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers[:] = []
        uvicorn_logger.propagate = True
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, DropNoisyAccess) for existing in access.filters):
        access.addFilter(DropNoisyAccess())
    # httpx logs every request at INFO; from a runner that is one line per poll.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
