"""Logging exists, renders JSON on demand, and stays quiet about runner polls."""

import json
import logging

from daikonstudio.logging import DropNoisyAccess, configure_logging


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
    assert len(logging.getLogger().handlers) == 1
    access = logging.getLogger("uvicorn.access")
    assert sum(isinstance(f, DropNoisyAccess) for f in access.filters) == 1
