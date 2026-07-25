"""Reserved workspace sentinel for shared, workspace-agnostic reference data."""

from __future__ import annotations

import uuid

# Reference data (organisms, proteins, proteomes) is shared across all tenants.
# It is stored under this reserved workspace id so it reuses all workspace-scoped
# machinery (events, audit, base repository) without special-casing.
GLOBAL_WORKSPACE_ID: uuid.UUID = uuid.UUID(int=0)
