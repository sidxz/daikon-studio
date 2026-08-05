"""Entrypoint: `python -m daikonstudio.infrastructure.runner`."""

from __future__ import annotations

import asyncio

from daikonstudio.infrastructure.runner import agent

if __name__ == "__main__":
    asyncio.run(agent.main())
