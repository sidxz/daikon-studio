"""Build identity for the running backend image.

Resolves version, git SHA and build date from the environment the image (CI)
or ``make dev`` (``scripts/build-info.sh``) baked in. ``APP_*`` is the shared
name family with the frontend. With nothing set every value is the honest dev
fallback; ``pyproject.toml`` is deliberately NOT consulted: it is a placeholder,
git tags are the source of truth (RELEASING.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEV_VERSION = "0.0.0+dev"


@dataclass(frozen=True)
class BuildInfo:
    """Identity of the running build."""

    version: str
    git_sha: str
    build_date: str


def build_info() -> BuildInfo:
    """Return the running build's identity."""
    return BuildInfo(
        version=os.environ.get("APP_VERSION") or DEV_VERSION,
        git_sha=os.environ.get("APP_GIT_SHA") or "unknown",
        build_date=os.environ.get("APP_BUILD_DATE") or "unknown",
    )
