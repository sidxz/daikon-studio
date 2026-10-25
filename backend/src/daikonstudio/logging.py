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
import time

import structlog

#: Paths whose *successful* access lines say nothing: two runners polling every
#: 3 s is 40 lines a minute. A failing poll or probe still logs.
NOISY_PATHS = ("/api/v1/runner/claim", "/health", "/ready")

# Uvicorn's access line is `<client> - "<method> <path> HTTP/<v>" <status>`, with
# the status code last and nothing after it -- so the match must accept end-of-line.
_SUCCESS = re.compile(r'" (200|204)(\s|$)')

#: How `configure_logging` recognises the handler it installed on a previous call.
_HANDLER_NAME = "daikonstudio"

#: How long one repeating traceback keeps its silence. The first copy carries the
#: news; the thousandth carries the same news at the price of a disk.
THROTTLE_WINDOW_SECONDS = 60.0

#: Distinct error signatures remembered at once -- a bound, so that a loop whose
#: messages vary cannot grow this dict instead of growing the log.
_THROTTLE_LIMIT = 512


class ThrottleRepeats(logging.Filter):
    """One copy of a repeating traceback per window, then a count of the rest.

    On 2026-10-22 a full Docker disk made Postgres crash-loop, which turned every
    runner poll into an `unhandled error` traceback: 5.1 GB written in nine hours,
    and rendering them saturated a core until the API answered nothing at all. The
    outage was caused by reporting the outage, so the report is what is bounded
    here.

    Only records carrying an exception are throttled: those are the expensive ones
    -- kilobytes where a line is bytes -- and a crash loop is what repeats them.
    The key holds the logger, the message and the exception type, so a *different*
    failure arriving mid-loop is still news and still gets through.
    """

    def __init__(self, window: float = THROTTLE_WINDOW_SECONDS) -> None:
        super().__init__()
        self._window = window
        self._seen: dict[tuple[str, str, str], tuple[float, int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        exception = record.exc_info[0] if record.exc_info else None
        if exception is None:
            return True
        key = (record.name, record.getMessage(), exception.__name__)
        now = time.monotonic()
        spoke_at, held = self._seen.get(key, (None, 0))
        if spoke_at is not None and now - spoke_at < self._window:
            self._seen[key] = (spoke_at, held + 1)
            return False
        if len(self._seen) >= _THROTTLE_LIMIT:
            self._seen.clear()
        self._seen[key] = (now, 0)
        if held:
            # Carried by the copy that gets through, so a reader learns the rate
            # and not merely that the thing happened again.
            record.msg = f"{record.getMessage()} [{held} more since]"
            record.args = None
        return True


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
        # Named explicitly because structlog's default is rich's formatter whenever
        # rich is importable -- and its `show_locals` defaults to True, so every
        # traceback dumped every frame's variables. asyncpg's include two full
        # connection-parameter reprs, password and all, which put the database
        # password in the log as often as the error.
        else structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(), exception_formatter=structlog.dev.plain_traceback
        )
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    # JSONRenderer is a bare json.dumps: without format_exc_info ahead of it a
    # `logger.exception` renders as `"exc_info": true` and the traceback is gone --
    # in exactly the format production runs. ConsoleRenderer formats exc_info itself.
    formatting: list[structlog.types.Processor] = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta
    ]
    if fmt == "json":
        formatting.append(structlog.processors.format_exc_info)
    formatting.append(renderer)
    handler = logging.StreamHandler(sys.stderr)
    handler.name = _HANDLER_NAME
    handler.addFilter(ThrottleRepeats())
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processors=formatting)
    )
    root = logging.getLogger()
    # Replace only the handler this function installed last time. A handler someone
    # else attached (pytest's caplog, a test harness, an APM agent) stays: wiping it
    # made every caplog assertion in the suite fail once any test had built the app.
    root.handlers[:] = [h for h in root.handlers if h.name != _HANDLER_NAME] + [handler]
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
