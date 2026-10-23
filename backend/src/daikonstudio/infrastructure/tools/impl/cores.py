"""The core-finding and R-group tools, over the vendored SAR chemistry."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

IMPLEMENTATIONS: dict[str, ToolImpl] = {}
