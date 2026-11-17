"""Logging exists, renders JSON on demand, and stays quiet about runner polls."""

import json
import logging
import sys
import time

from daikonstudio.logging import DropNoisyAccess, ThrottleRepeats, configure_logging


def _access(message: str) -> logging.LogRecord:
    return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, message, None, None)


def test_runner_polls_and_probes_are_dropped_from_the_access_log():
    noisy = DropNoisyAccess()
    # Uvicorn's real shape: the status code ends the line.
    assert not noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runner/claim HTTP/1.1" 204'))
    assert not noisy.filter(_access('127.0.0.1:1 - "GET /health HTTP/1.1" 200'))
    # And the shape with a reason phrase, which some formatters append.
    assert not noisy.filter(_access('127.0.0.1:1 - "GET /ready HTTP/1.1" 200 OK'))
    assert noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runs HTTP/1.1" 202'))
    # A failing poll is news.
    assert noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runner/claim HTTP/1.1" 500'))


def test_httpx_request_lines_are_below_the_threshold():
    configure_logging()
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING


def test_json_format_renders_stdlib_records_as_json(capsys):
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("daikonstudio.test").warning("hello %s", "world")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello world"
    assert record["level"] == "warning"
    assert record["logger"] == "daikonstudio.test"


def test_info_is_visible_once_configured(capsys):
    configure_logging(level="INFO", fmt="console")
    logging.getLogger("daikonstudio.test").info("realm scope active")
    assert "realm scope active" in capsys.readouterr().err


def test_configuring_twice_installs_one_handler_and_one_filter(capsys):
    configure_logging()
    configure_logging()
    ours = [h for h in logging.getLogger().handlers if h.name == "daikonstudio"]
    assert len(ours) == 1
    access = logging.getLogger("uvicorn.access")
    assert sum(isinstance(f, DropNoisyAccess) for f in access.filters) == 1


def test_handlers_someone_else_installed_survive(capsys):
    """pytest's caplog is a root handler. `create_app()` configures logging, so
    without this every caplog assertion after the first API test came back empty."""
    foreign = logging.NullHandler()
    foreign.name = "someone-else"
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        configure_logging()
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_json_format_carries_the_traceback(capsys):
    """Production logs JSON. A `logger.exception` there must carry the exception
    type, message and stack, not `"exc_info": true`."""
    configure_logging(level="INFO", fmt="json")
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("daikonstudio.test").exception("job failed")
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "job failed"
    assert "ValueError: boom" in record["exception"]
    assert "Traceback" in record["exception"]


def _failure(error: type[Exception] = ConnectionError) -> logging.LogRecord:
    try:
        raise error("connection was closed in the middle of operation")
    except error:
        return logging.LogRecord(
            "daikonstudio.test",
            logging.ERROR,
            __file__,
            1,
            "unhandled error",
            None,
            sys.exc_info(),
        )


def test_a_repeating_traceback_is_logged_once_per_window():
    """A crash-looping database wrote 5.1 GB of one traceback and pinned a core
    rendering it, so the API stopped answering. The first copy is the news."""
    throttle = ThrottleRepeats(window=60.0)
    assert throttle.filter(_failure())
    assert not throttle.filter(_failure())
    assert not throttle.filter(_failure())
    # A different failure arriving mid-loop is still news.
    assert throttle.filter(_failure(TimeoutError))
    # And a line with no exception is cheap, so it passes untouched.
    plain = logging.LogRecord("x", logging.INFO, __file__, 1, "job claimed", None, None)
    assert throttle.filter(plain)


def test_the_copy_that_gets_through_says_how_many_it_stands_for(monkeypatch):
    clock = iter([0.0, 1.0, 2.0, 100.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    throttle = ThrottleRepeats(window=60.0)
    assert throttle.filter(_failure())
    assert not throttle.filter(_failure())
    assert not throttle.filter(_failure())
    record = _failure()
    assert throttle.filter(record)
    assert "2 more since" in record.getMessage()


def test_a_console_traceback_does_not_dump_local_variables(capsys):
    """structlog reaches for rich's formatter whenever rich is importable, and its
    `show_locals` defaults to True: asyncpg's frames then put the database
    password in the log on every error, at kilobytes a copy."""
    configure_logging(level="INFO", fmt="console")
    try:
        password = "studio"  # noqa: F841 -- the point is that it must not be logged
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("daikonstudio.test").exception("unhandled error")
    written = capsys.readouterr().err
    assert "ValueError: boom" in written
    assert "password" not in written
