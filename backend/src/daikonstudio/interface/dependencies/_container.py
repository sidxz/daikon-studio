"""Reaching the DI container from a route.

Deliberately not in `_core.py`: `tests/api/test_auth_dependency.py` calls
`importlib.reload` on that module, which rebinds every function object it defines.
Anything a route captures by identity at import time -- which is what
`Depends(...)` does -- would then no longer match a later lookup of the same name.
`get_container` reads the container off `request.app.state`, so nothing here is
identity-sensitive either way, and keeping it out of the reloaded module means it
stays that way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from lagom import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def use_case[T](use_case_type: type[T]) -> Callable[[Container], T]:
    def _dependency(container: Annotated[Container, Depends(get_container)]) -> T:
        return container[use_case_type]

    return _dependency
