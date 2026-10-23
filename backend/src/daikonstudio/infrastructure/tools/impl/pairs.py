"""The matched-pair tools, over the R-group table."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

IMPLEMENTATIONS: dict[str, ToolImpl] = {}
