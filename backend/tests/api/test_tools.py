"""The tools catalogue over HTTP.

Read-only and workspace-agnostic: the catalogue is a property of the deployment,
not of a workspace, so every signed-in caller sees the same list. It still needs
authentication, because an unauthenticated caller should learn nothing about what
this server can do.
"""

from __future__ import annotations

import httpx


async def test_lists_tools_in_catalogue_order(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) > 5
    ids = [tool["id"] for tool in tools]
    assert ids.index("standardize") < ids.index("decompose_rgroups"), (
        "Catalogue order is presentation order: a chemist standardizes before decomposing."
    )
    first = tools[0]
    assert set(first) == {
        "id",
        "version",
        "name",
        "description",
        "area",
        "inputs",
        "outputs",
        "library",
        "judgment",
        "resolves_to",
        "surface",
        "external",
        "expands_to",
    }


async def test_a_tool_declares_what_it_consumes_and_produces(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/tools/decompose_rgroups")
    assert response.status_code == 200
    tool = response.json()
    assert {slot["kind"] for slot in tool["inputs"]} == {"compound_table", "core"}
    assert [slot["kind"] for slot in tool["outputs"]] == ["rgroup_table"]
    assert tool["surface"] == "view"


async def test_a_tool_that_needs_a_chemist_says_so(client: httpx.AsyncClient) -> None:
    """A client must not present the first proposed core as decided."""
    response = await client.get("/api/v1/tools/find_cores")
    assert response.status_code == 200
    tool = response.json()
    assert tool["judgment"] is True
    assert tool["surface"] == "editor"
    assert tool["external"] is True
    # A chemist's choice among these candidates is what produces a core. No tool
    # does, which is why a study can record who approved it.
    assert tool["resolves_to"] == "core"


async def test_unknown_tool_is_404_not_500(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/tools/no-such-tool")
    assert response.status_code == 404
    assert "no-such-tool" in response.json()["message"]


async def test_catalogue_needs_authentication(anonymous_client: httpx.AsyncClient) -> None:
    response = await anonymous_client.get("/api/v1/tools")
    assert response.status_code == 401


async def test_a_composite_names_the_sequence_it_stands_for(client: httpx.AsyncClient) -> None:
    """One catalogue entry for a sequence everyone runs together, so the planner
    makes one choice instead of several it could order wrongly."""
    response = await client.get("/api/v1/tools/build_sar_table")
    assert response.status_code == 200
    tool = response.json()
    assert tool["expands_to"] == ["decompose_rgroups", "name_substituents"]
    assert tool["area"] == "sar"


async def test_tools_are_grouped_by_area(client: httpx.AsyncClient) -> None:
    areas = {tool["area"] for tool in (await client.get("/api/v1/tools")).json()}
    assert areas == {"data", "sar", "context"}
