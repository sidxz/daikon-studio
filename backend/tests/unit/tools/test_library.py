"""The six set-level RDKit tools, on a table with a deliberate unparseable row.

One table serves every test: three real drugs, one acyclic molecule and one string that
is not a structure. Every tool has to survive the last two, since a real upload has both.
"""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.infrastructure.tools.impl import library
from daikonstudio.infrastructure.tools.params import (
    ClusterParams,
    DiverseParams,
    SimilarityParams,
    SubstructureParams,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC = "O=C(O)c1ccccc1O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
ACYCLIC = "CCCCO"
BROKEN = "not a molecule"


@pytest.fixture
def table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "smiles": [ASPIRIN, SALICYLIC, IBUPROFEN, ACYCLIC, BROKEN],
        }
    )


def test_substructure_search_flags_rows_and_nulls_the_unparseable(table: pl.DataFrame) -> None:
    out = library.substructure_search(SubstructureParams(smarts="c1ccccc1"), {"compounds": table})
    assert out["compounds"]["matches_query"].to_list() == [True, True, True, False, None]
    assert out["meta"]["n_matched"] == 3
    assert out["meta"]["n_unparsed"] == 1
    # The table is annotated, never filtered: a search that matched nothing must still
    # return the rows, so a zero reads as a result rather than a broken step.
    assert out["compounds"].height == table.height


def test_a_bad_smarts_names_itself(table: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="valid SMARTS"):
        library.substructure_search(SubstructureParams(smarts="c1ccccc1("), {"compounds": table})


def test_similarity_search_ranks_the_closest_first(table: pl.DataFrame) -> None:
    out = library.similarity_search(SimilarityParams(query_smiles=ASPIRIN), {"compounds": table})
    ranked = out["compounds"]
    assert ranked["id"][0] == "a"  # the query itself
    assert ranked["tanimoto"][0] == pytest.approx(1.0)
    # Salicylic acid is aspirin minus the acetyl, so it must outrank ibuprofen.
    order = ranked["id"].to_list()
    assert order.index("b") < order.index("c")
    assert ranked["tanimoto"][-1] is None  # the unparseable row sorts last, never first
    assert out["meta"]["best"] == pytest.approx(1.0)


def test_clustering_numbers_from_the_largest_and_marks_one_centroid_each(
    table: pl.DataFrame,
) -> None:
    out = library.cluster_compounds(ClusterParams(cutoff=0.6), {"compounds": table})
    frame = out["compounds"]
    assert frame["cluster"][4] is None  # unparseable rows join no cluster
    assert frame["is_centroid"][4] is None
    numbers = [c for c in frame["cluster"].to_list() if c is not None]
    assert len(numbers) == 4
    assert min(numbers) == 1
    # Exactly one centroid per cluster.
    centroids = sum(1 for c in frame["is_centroid"].to_list() if c)
    assert centroids == out["meta"]["n_clusters"]


def test_the_diverse_pick_is_a_subset_and_repeats_exactly(table: pl.DataFrame) -> None:
    first = library.pick_diverse_subset(DiverseParams(n=3), {"compounds": table})
    again = library.pick_diverse_subset(DiverseParams(n=3), {"compounds": table})
    assert first["compounds"]["id"].to_list() == again["compounds"]["id"].to_list()
    assert first["meta"] == {"requested": 3, "picked": 3, "from_rows": 4, "n_unparsed": 1}
    assert BROKEN not in first["compounds"]["smiles"].to_list()
    # Asking for more than the table holds returns what there is, not an error.
    everything = library.pick_diverse_subset(DiverseParams(n=99), {"compounds": table})
    assert everything["meta"]["picked"] == 4


def test_scaffolds_separate_an_acyclic_compound_from_an_unparseable_one(
    table: pl.DataFrame,
) -> None:
    out = library.murcko_scaffolds(None, {"compounds": table})
    scaffolds = out["compounds"]["scaffold"].to_list()
    assert scaffolds[0] == "c1ccccc1"
    assert scaffolds[3] == ""  # acyclic: no scaffold, which is a finding
    assert scaffolds[4] is None  # unparseable: no answer at all
    assert out["meta"]["n_acyclic"] == 1
    assert out["meta"]["n_distinct_scaffolds"] == 1
    assert out["compounds"]["scaffold_generic"][0] == "C1CCCCC1"
    # The ids carry the grouping the agent needs, since the strings are withheld from it.
    # The three benzene-scaffold compounds share id 1; the acyclic and unparseable rows have
    # none, because neither belongs to a chemotype.
    assert out["compounds"]["scaffold_id"].to_list() == [1, 1, 1, None, None]


def test_synthetic_accessibility_scores_in_range_and_says_what_it_is_not(
    table: pl.DataFrame,
) -> None:
    out = library.synthetic_accessibility(None, {"compounds": table})
    scores = out["properties"]["sa_score"].to_list()
    assert scores[4] is None
    for score in scores[:4]:
        assert score is not None and 1.0 <= score <= 10.0
    assert "not a route" in out["meta"]["note"]


def test_scaffold_columns_are_withheld_from_the_agent(table: pl.DataFrame) -> None:
    """A bare ring such as c1ccccc1 has no bond, branch or bracket mark, so the content
    test cannot see it as a structure. Scaffold columns are withheld by name instead."""
    from daikonstudio.application.tools import codecs

    out = library.murcko_scaffolds(None, {"compounds": table})
    _, summary = codecs.encode("compound_table", out["compounds"])
    assert "scaffold" in summary["withheld"]
    assert "scaffold_generic" in summary["withheld"]
    # The ids must survive: they are what lets the agent compare groupings it cannot see.
    assert "scaffold_id" not in summary["withheld"]
    assert "scaffold_generic_id" not in summary["withheld"]


def test_the_answer_survives_a_table_too_long_to_look_at() -> None:
    """The agent is shown a summary and may ask for 50 rows from the top. On a long table
    the rows that answer the question are not in those 50, so `meta` has to name them."""
    rows = [{"compound_id": f"X{i:04d}", "smiles": ASPIRIN} for i in range(200)]
    rows += [{"compound_id": f"N{i:03d}", "smiles": "O=[N+]([O-])c1ccccc1"} for i in range(30)]
    frame = pl.DataFrame(rows)

    out = library.substructure_search(
        SubstructureParams(smarts="[N+](=O)[O-]"), {"compounds": frame}
    )
    # Not one match is inside the first 50 rows.
    assert not any(r["matches_query"] for r in out["compounds"].head(50).to_dicts())
    assert out["meta"]["n_matched"] == 30
    assert out["meta"]["matched"]["ids"][0] == "N000"
    assert out["meta"]["matched"]["truncated"] is True  # 30 named of a 25 cap

    clustered = library.cluster_compounds(ClusterParams(), {"compounds": frame})
    named = {c["centroid"] for c in clustered["meta"]["clusters"]}
    assert any(name.startswith("N") for name in named), named


def test_every_tool_is_reachable_through_the_runtime_registry() -> None:
    from daikonstudio.infrastructure.tools.catalog import default_registry
    from daikonstudio.infrastructure.tools.runtime import IMPLEMENTATIONS

    declared = {
        "substructure_search",
        "similarity_search",
        "cluster_compounds",
        "pick_diverse_subset",
        "murcko_scaffolds",
        "synthetic_accessibility",
    }
    catalogued = {tool.id for tool in default_registry().manifests()}
    assert declared <= catalogued, declared - catalogued
    assert declared <= set(IMPLEMENTATIONS), declared - set(IMPLEMENTATIONS)


def test_a_table_tools_findings_reach_the_agent(table: pl.DataFrame) -> None:
    """A table has nowhere in its blob to carry what the tool found, so `meta` is folded
    into the summary. Without this the counts and the named rows are computed and binned."""
    from daikonstudio.application.studies import readers
    from daikonstudio.application.tools.codecs import encode
    from daikonstudio.infrastructure.tools.runtime import run_tool

    blob, _ = encode("compound_table", table)
    outputs = run_tool("substructure_search", {"smarts": "c1ccccc1"}, {"compounds": blob})

    assert outputs[0].summary["meta"]["n_matched"] == 3
    told = readers.summarize_table(None, outputs[0].summary)
    assert "n_matched" in told  # the agent is told the finding, not only the shape
