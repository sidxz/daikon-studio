"""Opening a study, attaching files to it, and starting it.

What these pin is the door. The first message is enough to make a study; a file is read into
it once and by kind; and whatever is refused leaves nothing behind: no study, no value, no
blob in the store. Starting reads the plan against what has been brought, so a plan drafted
before its file is not started without it, and is started once the file has come.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import polars as pl
from returns.result import Result

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.planner.plan import Interrogation, Question
from daikonstudio.application.studies.agent import StudyAgent
from daikonstudio.application.studies.open_study import (
    AttachFiles,
    OpenCommand,
    OpenStudy,
    Upload,
    bring,
)
from daikonstudio.application.studies.standing import standing
from daikonstudio.application.studies.start_study import StartStudy
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.structures import describe_pdb
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ValidationError
from daikonstudio.domain.studies.study import Step, Study
from daikonstudio.infrastructure.tables.profiler import PolarsTableProfiler
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.llm import ScriptedLanguageModel, ScriptedLanguageModels
from tests.fakes.studies import InMemoryStudies

REGISTRY = default_registry()
AUTH = FakeAuth()
GOAL = "Which substituents move potency in this series?"
RESTATED = "Which substituents move potency, in the compounds sent."
# One reading is the text "NaN", as an export from numpy or R writes a missing one.
COMPOUNDS = b"smiles,pIC50\nCCO,5.1\nCCN,NaN\nc1ccccc1,7.0\n"
UBIQUITIN = (Path(__file__).parents[2] / "fixtures" / "1ubq.pdb").read_bytes()
PROFILE_HEADER = "The uploaded file has 3 rows"


async def _no_run(
    study_id: uuid.UUID,
    auth: AuthContext | None = None,
    *,
    start: uuid.UUID | None = None,
    until: uuid.UUID | None = None,
) -> Result[Study, DomainError]:
    raise AssertionError("opening a study runs nothing")


class World:
    """The services over in-memory stores, with the one question the model asks scripted."""

    def __init__(self) -> None:
        question = Question(ask="Which column is the readout?", why="It decides the analysis.")
        self.studies = InMemoryStudies()
        self.store = InMemoryBlobStore()
        self.model = ScriptedLanguageModel(
            Interrogation(restated_goal=RESTATED, questions=[question])
        )
        profiler = PolarsTableProfiler()
        agent = StudyAgent(
            self.studies, REGISTRY, ScriptedLanguageModels(self.model), self.store, _no_run
        )
        self.open = OpenStudy(self.studies, self.store, profiler, agent)
        self.attach = AttachFiles(self.studies, self.store, profiler, REGISTRY)
        self.start = StartStudy(self.studies, REGISTRY)

    def draft(self, *plan: str) -> Study:
        """A stored draft whose steps are the tools named, as a planning turn leaves one."""
        steps = []
        for position, tool_id in enumerate(plan):
            manifest = REGISTRY.get(tool_id)
            steps.append(
                Step(
                    id=uuid.uuid4(),
                    position=position,
                    tool_id=tool_id,
                    tool_version=manifest.version,
                    name=manifest.name,
                    purpose="p",
                    params={},
                    gated=False,
                    always=manifest.automatic,
                    expanded_from=None,
                )
            )
        study = Study(
            workspace_id=AUTH.workspace_id,
            created_by=AUTH.user_id,
            goal=GOAL,
            restated_goal="",
            checks=(),
            stop_conditions=(),
            steps=steps,
        )
        self.studies.rows[study.id] = study
        return study


async def test_opening_brings_the_files_and_asks_first() -> None:
    world = World()
    files = (Upload("series.csv", COMPOUNDS), Upload("1ubq.pdb", UBIQUITIN))

    opened = (await world.open(OpenCommand(goal=GOAL, files=files), AUTH)).unwrap()

    stored = world.studies.rows[opened.id]
    brought = {value.kind: value for value in stored.values}
    assert set(brought) == {"compound_table", "structure"}
    assert all(v.produced_by is None and v.key == v.kind for v in stored.values)
    table, structure = brought["compound_table"], brought["structure"]
    profile = table.summary["profile"]
    assert profile["rows"] == 3
    assert [column["name"] for column in profile["columns"]] == ["smiles", "pIC50"]
    rows = codecs.decode("compound_table", world.store.get_bytes(table.blob_key))
    assert isinstance(rows, pl.DataFrame) and rows.height == 3
    kept = codecs.decode("structure", world.store.get_bytes(structure.blob_key))
    assert isinstance(kept, dict) and kept["format"] == "pdb"
    assert kept["meta"] == structure.summary
    assert structure.summary["chains"] == ["A"]
    # The study row is JSON, which has no NaN: the profiler read that "NaN" as a number.
    json.dumps([value.summary for value in stored.values], allow_nan=False)

    (turn,) = stored.thread.turns
    assert (turn.role, turn.kind, turn.say) == ("agent", "asking", RESTATED)
    assert [q.ask for q in turn.questions] == ["Which column is the readout?"]
    assert stored.restated_goal == RESTATED
    assert stored.steps == [] and stored.phase == "planning"
    # What the model asked its question from: the table's shape, not only the goal.
    assert PROFILE_HEADER in world.model.calls[0][1]


async def test_a_second_table_is_refused_and_nothing_is_created() -> None:
    world = World()
    files = (Upload("round1.csv", COMPOUNDS), Upload("round2.tsv", COMPOUNDS.replace(b",", b"\t")))

    result = await world.open(OpenCommand(goal=GOAL, files=files), AUTH)

    error = result.failure()
    assert isinstance(error, ValidationError)
    assert error.message == "A study takes one compound table file."
    assert world.studies.rows == {}
    assert world.store.blobs == {}  # the first file's blob was taken back with the refusal
    assert world.model.calls == []


async def test_a_cif_or_an_unreadable_csv_is_refused_with_a_sentence() -> None:
    def opening(*files: Upload, goal: str = GOAL) -> OpenCommand:
        return OpenCommand(goal=goal, files=files)

    refused = [
        (opening(Upload("model.cif", b"data_1UBQ")), "That file type is not supported."),
        (
            opening(Upload("ragged.csv", b"a,b\n1,2,3\n4,5\n")),
            "The compounds file could not be read.",
        ),
        (opening(Upload("empty.csv", b"")), "The compounds file could not be read."),
        (opening(Upload("empty.pdb", b"")), "The structure file could not be read."),
        (opening(goal="Too short."), "Say what you want to find out."),
    ]
    errors = []
    for command, sentence in refused:
        world = World()

        error = (await world.open(command, AUTH)).failure()

        assert isinstance(error, ValidationError) and error.message == sentence, sentence
        assert world.studies.rows == {} and world.store.blobs == {}, sentence
        errors.append(error)
    # The one a scientist is likeliest to try says what to attach instead.
    assert errors[0].detail == "Attach a .csv, .tsv or .pdb file; .cif is not read yet."


async def test_start_needs_the_compounds_file_the_plan_hands_over() -> None:
    world = World()
    study = world.draft("predict_admet")

    refused = (await world.start(study.id, AUTH)).failure()

    assert isinstance(refused, ValidationError)
    assert refused.message.startswith("Attach the compounds file")
    assert world.studies.rows[study.id].started_at is None

    (await world.attach(study.id, (Upload("series.csv", COMPOUNDS),), AUTH)).unwrap()
    started = (await world.start(study.id, AUTH)).unwrap()

    assert world.studies.rows[study.id].started_at is not None
    # Standardizing is placed once there is a table to standardize, ahead of the step.
    assert [s.tool_id for s in started.steps] == ["standardize", "predict_admet"]


async def test_start_refuses_a_plan_with_a_problem_and_records_the_event() -> None:
    world = World()
    study = world.draft("find_pockets")  # no structure fetched, none brought

    refused = (await world.start(study.id, AUTH)).failure()

    assert isinstance(refused, ConflictError)
    pockets = REGISTRY.get("find_pockets").name
    assert refused.message.startswith(f"The plan cannot start yet: Step 1 ({pockets}) needs")
    assert world.studies.rows[study.id].started_at is None
    assert world.studies.rows[study.id].thread.turns == []

    # The file the plan was waiting for comes. Before anything is pressed, the plan on the
    # page has the preparation the file makes possible, and the problem is gone.
    (await world.attach(study.id, (Upload("1ubq.pdb", UBIQUITIN),), AUTH)).unwrap()

    attached = world.studies.rows[study.id]
    assert [s.tool_id for s in attached.steps] == ["prepare_structure", "find_pockets"]
    assert all(s.problem is None for s in attached.steps)

    started = (await world.start(study.id, AUTH)).unwrap()

    stored = world.studies.rows[study.id]
    assert stored.started_at is not None and started.phase == "running"
    events = [(t.role, t.kind, t.say) for t in stored.thread.turns]
    assert events == [
        ("event", "attached", "Attached 1ubq.pdb as structure."),
        ("event", "started", "Started."),
    ]


def test_describe_pdb_lists_chains_and_ligands() -> None:
    text = UBIQUITIN.decode()

    assert describe_pdb(text) == {"id": None, "source": "upload", "chains": ["A"], "ligands": []}

    # A zinc 0.9 angstrom from the first alpha carbon, in the fixed columns a PDB uses.
    zinc = "HETATM  603 ZN    ZN A 100      27.000  25.000   3.000  1.00 20.00          ZN  "
    (ligand,) = describe_pdb(f"{text}\n{zinc}")["ligands"]
    assert (ligand["comp_id"], ligand["chain"], ligand["resseq"]) == ("ZN", "A", 100)
    assert {"chain": "A", "resname": "MET", "resseq": 1} in ligand["contacts"]


async def test_the_profile_is_shown_while_drafting_and_not_once_started() -> None:
    world = World()
    study = world.draft("predict_admet")
    await bring(
        study, Upload("series.csv", COMPOUNDS), store=world.store, profiler=PolarsTableProfiler()
    )

    drafting = standing(study, REGISTRY)
    study.start()
    running = standing(study, REGISTRY)

    assert PROFILE_HEADER in drafting
    # The table is still listed as brought; only its profile is a planning-time view.
    assert "compound table (3 rows, 2 columns)" in running and PROFILE_HEADER not in running
