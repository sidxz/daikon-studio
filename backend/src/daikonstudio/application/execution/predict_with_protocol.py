"""Prediction runs: the other half of publishing.

Training produces a Protocol only its author has ever touched; this module is
what makes a *published* one a shared asset -- a colleague uploads their own
compounds and runs someone else's Protocol against them, without ever seeing
the training Dataset or the fitted weights directly. `PredictWithProtocol`
creates the Run (or hands back an already-computed one); `RunPrediction` is
the `RunKind.PREDICTION` worker handler that actually scores the compounds,
mirroring `train_protocol.py`'s `TrainProtocol`/`RunTraining` split -- request
handling and worker execution live together because they are the only two
places that know a prediction Run's `params` shape.

Three decisions, made deliberately:

1. **Uncertainty.** XGBoost has no ensemble spread to report (see
   `_predict_with_tree_ensemble`'s own docstring) and returns `None`. That
   `None` is carried straight through to `PredictionRow.uncertainty` rather
   than being coerced to `0.0` -- a fabricated number would be read by a
   triage grid as "the model is confident here", which is worse than an
   admitted "not available".
2. **Applicability.** A continuous nearest-neighbour Tanimoto similarity to
   the Protocol's own training set (`PredictionRow.applicability`), not a
   boolean and not a fixed in/out-of-domain flag. It is the exact metric and
   threshold (0.3) `build_scorecard.py`'s `applicability_coverage` already
   uses, sourced from the same `ScorecardInputs.train_structures` a Protocol's
   training run wrote -- so a compound this screen calls out-of-domain and one
   the Scorecard's coverage number excludes are always the same compound. A
   raw number, not a pre-thresholded flag, is what lets a triage grid filter
   at 0.3 or at any other cut without a second call.
3. **Cancellation and caching.** `find_by_cache_key` returns the newest Run
   for a `(workspace_id, cache_key)` pair regardless of status -- a FAILED or
   CANCELLED Run shares its cache_key with every future identical request.
   Neither is a result to serve: a stored failure replayed as data is worse
   than paying for a fresh run, and a cancelled request was never allowed to
   finish in the first place. Both fall through to a brand-new Run by the
   same mechanism -- `PredictWithProtocol` only ever treats a `READY` hit as
   reusable.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.catalog.chemical_space import (
    NEIGHBOURS,
    neighbours_key,
    neighbours_parquet,
)
from daikonstudio.application.data.create_dataset import upload_key
from daikonstudio.application.data.prepare_frame import read_csv_upload
from daikonstudio.application.engines.context import PredictContext
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.result_view import (
    ROW_ID,
    RangeFilter,
    SortSpec,
    apply_result_view,
)
from daikonstudio.application.execution.train_protocol import ScorecardInputs, scorecard_inputs_key
from daikonstudio.application.pagination import PageResult, clamp_limit
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)


def predictions_key(workspace_id: uuid.UUID, run_id: uuid.UUID) -> str:
    """Where one prediction Run's per-row results live.

    Addressed by *run*, not by protocol, unlike `scorecard_inputs_key`: a
    Scorecard is one honest answer per Protocol, but a published Protocol can
    be run against any number of different compound sets, and each run's rows
    belong to the Run that produced them.
    """
    return f"{workspace_id}/runs/{run_id}/predictions.parquet"


@dataclass(frozen=True, kw_only=True)
class PredictWithProtocolCommand:
    protocol_id: uuid.UUID
    upload_ref: str
    structure_column: str
    conditions: dict[str, Any] = field(default_factory=dict)
    # Which uploaded column names the compound (a registry id, a plate well).
    # Carried verbatim into the results as `compound_id` so a scientist can join
    # predictions back to the file they uploaded -- the one thing a canonical
    # SMILES cannot do for them. Optional: a file of bare structures has none.
    id_column: str | None = None

    def to_params(self) -> dict[str, Any]:
        return {
            "protocol_id": str(self.protocol_id),
            "upload_ref": self.upload_ref,
            "structure_column": self.structure_column,
            "conditions": self.conditions,
            "id_column": self.id_column,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PredictWithProtocolCommand:
        # Explicit field-by-field, not `cls(**params)`: a frozen dataclass built
        # that way (see `ScorecardInputs.from_json`) raises `TypeError` the day a
        # new field is added and an old Run's stored params don't have it yet.
        # `.get()` with a default is what keeps this read path forward-compatible.
        return cls(
            protocol_id=uuid.UUID(params["protocol_id"]),
            upload_ref=params["upload_ref"],
            structure_column=params["structure_column"],
            conditions=params.get("conditions", {}),
            id_column=params.get("id_column"),
        )


class PredictWithProtocol:
    """Creates a prediction Run, or hands back a cached one -- the enqueuing
    half. `RunPrediction`, below, is the worker's other half.

    Only a *published* Protocol may be run: a draft is the author's own work
    in progress, and a model nobody else can run is not the shared asset
    publishing exists to create. Checked here, synchronously, alongside the
    Protocol's existence -- both are registry-membership-shaped questions with
    exactly one possible answer, the same reasoning `TrainProtocol` uses for
    rejecting an unknown engine before a Run is ever created. Whether the
    uploaded compounds parse, and whether `structure_column` is actually a
    column in them, is deferred to the worker instead: that mirrors how
    `TrainProtocol` defers condition validation, and it means a cache hit
    never pays for CSV parsing it doesn't need.
    """

    def __init__(
        self,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        enqueuer: JobEnqueuer,
        engines: EngineRegistry,
    ) -> None:
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._enqueuer = enqueuer
        self._engines = engines

    async def __call__(
        self, command: PredictWithProtocolCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        # Workspace-scoped exactly like every other Protocol read (Task 16): a
        # Protocol from another workspace does not exist as far as this call is
        # concerned. "Colleague" means another user of the *same* workspace --
        # publishing shares a Protocol across the people in it, not across tenants.
        protocol = await self._protocols.get(auth.workspace_id, command.protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(command.protocol_id)))
        if not protocol.is_locked:
            return Failure(
                ConflictError(
                    "Only a published protocol can be used for prediction. "
                    "Publish this protocol first."
                )
            )

        try:
            upload_ref = uuid.UUID(command.upload_ref)
        except ValueError:
            return Failure(
                ValidationError("The upload reference is invalid. Upload the file again.")
            )
        key = upload_key(auth.workspace_id, upload_ref)
        if not self._store.exists(key):
            return Failure(NotFoundError("Upload", str(upload_ref)))

        # The content itself, not just its ref: two different uploads could
        # collide on a ref only across workspaces (refs are per-workspace
        # UUIDs), which cache_key's own workspace scoping already prevents --
        # this hash is what makes the SAME bytes uploaded under a fresh ref
        # still count as identical work.
        #
        # ponytail: reads the whole upload into memory to hash it, on every
        # request including a cache hit. Bounded today by the datasets route's
        # MAX_UPLOAD_BYTES ceiling (the same `/uploads` endpoint predictions
        # reuse); upgrade path is a streaming/chunked sha256 if uploads ever
        # get large enough for this to matter.
        input_hash = hashlib.sha256(self._store.get_bytes(key)).hexdigest()
        cache_key = compute_cache_key(
            kind="prediction",
            protocol_id=str(protocol.id),
            protocol_version=protocol.protocol_version,
            input_hash=input_hash,
            # Which column is read as structures changes what gets predicted just
            # as much as the bytes or the conditions do -- two requests against
            # the identical upload that name different columns are different work
            # and must not collide on the same cache_key.
            structure_column=command.structure_column,
            conditions=command.conditions,
            # A different identifier column is a different results file.
            id_column=command.id_column,
        )

        # Only a READY hit is reusable. `find_by_cache_key` does not filter by
        # status -- a FAILED or CANCELLED Run sharing this cache_key must fall
        # through to a fresh Run below, never be handed back as though it held
        # a result (see this module's docstring, Decision 3).
        existing = await self._runs.find_by_cache_key(auth.workspace_id, cache_key)
        if existing is not None and existing.status is RunStatus.READY:
            return Success(existing)

        # Resolved before the Run row exists, so a Protocol whose engine this deployment
        # no longer ships fails as a clean 404 rather than leaving an orphan PENDING Run
        # that no worker can ever serve. Deliberately after the cache check: an already
        # READY result stays reusable even if the engine has since been removed.
        try:
            lane = self._engines.get(protocol.engine_id).manifest().lane
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", protocol.engine_id))

        run = Run(
            kind=RunKind.PREDICTION,
            workspace_id=auth.workspace_id,
            requested_by=auth.user_id,
            cache_key=cache_key,
            params=command.to_params(),
            # Also in `params`, which is what the worker reads to do the work.
            # Set here as well so the *column* answers "which Protocol is this
            # Run about" for both kinds -- a training Run has no such params
            # key, and a client should not have to know which kind it is
            # holding to find that out.
            protocol_id=command.protocol_id,
        )
        await self._runs.add(run)
        await self._enqueuer.enqueue(run.id, lane=lane)
        return Success(run)


class RunPrediction:
    """The `RunKind.PREDICTION` handler: score a colleague's own compounds
    against a published Protocol's frozen artifact.

    Deliberately does not depend on `DatasetRepository`: which column holds
    the structures comes from the request (`structure_column`), not from the
    training Dataset, so a prediction never has to reach back into training
    data that may since have been reused, renamed or is simply in a different
    workspace's history than the Protocol's current runner.
    """

    def __init__(
        self,
        protocols: ProtocolRepository,
        store: BlobStore,
        engines: EngineRegistry,
        normalizer: StructureNormalizer,
    ) -> None:
        self._protocols = protocols
        self._store = store
        self._engines = engines
        self._normalizer = normalizer

    async def __call__(self, run: Run) -> str:
        command = PredictWithProtocolCommand.from_params(run.params)
        protocol = await self._protocols.get(run.workspace_id, command.protocol_id)
        if protocol is None:
            raise NotFoundError("Protocol", str(command.protocol_id))

        raw = self._store.get_bytes(upload_key(run.workspace_id, uuid.UUID(command.upload_ref)))
        frame = read_csv_upload(raw)
        if command.structure_column not in frame.columns:
            raise ValidationError(
                f"Column '{command.structure_column}' is not in the uploaded file. "
                f"Available columns: {', '.join(frame.columns)}."
            )

        raw_structures = [str(value) for value in frame[command.structure_column].to_list()]
        canonical = [self._normalizer.canonicalize(smiles) for smiles in raw_structures]
        keep = [smiles is not None for smiles in canonical]
        is_valid = pl.Series(keep)
        valid_frame = frame.filter(is_valid).with_columns(
            pl.Series(command.structure_column, [s for s in canonical if s is not None])
        )
        if valid_frame.height == 0:
            raise ValidationError("No SMILES in the uploaded file could be parsed.")

        # 1-based data-row positions in the uploaded file, the same convention
        # `InvalidRow.row_number` uses. A dropped (unparseable) row leaves a gap,
        # which is how a scientist learns *which* five of 9,975 went missing.
        input_rows = [index + 1 for index, kept in enumerate(keep) if kept]
        compound_ids: list[str | None] | None = None
        if command.id_column is not None:
            if command.id_column not in frame.columns:
                raise ValidationError(
                    f"Identifier column '{command.id_column}' is not in the uploaded file. "
                    f"Available columns: {', '.join(frame.columns)}."
                )
            # Verbatim text, nullable, never a key: blanks become null and
            # duplicates both survive, because the file is the scientist's.
            raw_ids = frame[command.id_column].cast(pl.String, strict=False).to_list()
            compound_ids = [
                None if value is None or not value.strip() else value.strip()
                for value, kept in zip(raw_ids, keep, strict=True)
                if kept
            ]

        engine = self._engines.get(protocol.engine_id)
        # Read the artifact back from the URI the aggregate itself carries,
        # not a key re-derived from `protocol.id`: a versioned Protocol (Task
        # 12's `new_version()`) can have an `artifact_uri` written under its
        # *parent's* id when no retraining has produced a new one yet. Task 17
        # review, Important 3 -- re-deriving the key silently 404s that case.
        artifact = self._store.get_bytes(protocol.artifact_uri)
        predictions = await asyncio.to_thread(
            engine.predict,
            PredictContext(
                frame=valid_frame,
                structure_column=command.structure_column,
                artifact=artifact,
                conditions=command.conditions,
            ),
        )

        structures = valid_frame[command.structure_column].to_list()
        train_structures = self._train_structures(protocol)
        neighbours: tuple[list[list[int]], list[list[float]]] | None = None
        similarities: list[float | None]
        if structures and train_structures:
            neighbours = self._normalizer.nearest_neighbours_tanimoto(
                structures, train_structures, NEIGHBOURS
            )
            # Applicability is the nearest neighbour from the same search, so the
            # map, the triage column and the domain filter can never disagree.
            similarities = [row[0] for row in neighbours[1]]
        else:
            # Never a fabricated 0.0: an empty (or unreadable -- see
            # `_train_structures`) training set means "unmeasurable", not
            # "confirmed far from everything" (same reasoning as
            # `build_scorecard.py`'s own guard).
            similarities = [None] * len(structures)

        values = predictions["value"].to_list()
        # "structure"/"uncertainty"/"applicability" below, and the readout
        # name(s) they sit alongside, are exactly what
        # `domain.data.target.RESERVED_TARGET_COLUMNS` reserves against a
        # TargetSpec (C1, whole-branch review): a target sharing one of these
        # literal names would have this dict's later write silently overwrite
        # the earlier one. If this dict ever grows another literal key, add
        # it to that set too.
        columns: dict[str, pl.Series] = {"structure": pl.Series(structures)}
        columns["input_row"] = pl.Series(input_rows, dtype=pl.Int64)
        if compound_ids is not None:
            columns["compound_id"] = pl.Series(compound_ids, dtype=pl.String)
        if len(protocol.readouts) == 1:
            columns[protocol.readouts[0].name] = pl.Series(values, dtype=pl.Float64)
        else:
            # Classification: `derive_readouts` always orders these
            # (probability, class). `value` is P(class=1); the hard label is
            # the standard 0.5 decision threshold over it -- the engine's own
            # `predict()` only ever returns the probability (see
            # `_scoring.py`), so this is the one place a class label exists.
            #
            # ponytail: 0.5 is fixed, not configurable -- there is nowhere for
            # a scientist to ask for a different operating point (e.g. to
            # trade recall for precision on an imbalanced assay). Upgrade
            # path: accept it as a prediction condition once someone needs one.
            probability_readout, class_readout = protocol.readouts
            columns[probability_readout.name] = pl.Series(values, dtype=pl.Float64)
            columns[class_readout.name] = pl.Series(
                [1.0 if v >= 0.5 else 0.0 for v in values], dtype=pl.Float64
            )
        columns["uncertainty"] = predictions["uncertainty"]
        columns["applicability"] = pl.Series(similarities, dtype=pl.Float64)

        # Rides out on run_job's own `succeed()` + `update()`, like a training
        # run's headline metric -- a run is never READY without its counts.
        run.record_prediction_counts(uploaded_rows=frame.height, scored_rows=valid_frame.height)

        buffer = io.BytesIO()
        pl.DataFrame(columns).write_parquet(buffer)
        result_uri = self._store.put_bytes(
            predictions_key(run.workspace_id, run.id), buffer.getvalue()
        )
        if neighbours is not None:
            # Beside the results, never inside them: the map reads this, and a run's
            # results file is written exactly once.
            self._store.put_bytes(
                neighbours_key(run.workspace_id, run.id), neighbours_parquet(*neighbours)
            )
        return result_uri

    def _train_structures(self, protocol: InSilicoProtocol) -> list[str] | None:
        """The Protocol's own training set, read from the same
        `ScorecardInputs` blob its Scorecard reads (Task 15/16) -- the single
        definition of "what this model was trained on", so applicability here
        and the Scorecard's coverage number can never disagree about which
        compounds are in-domain.

        Best-effort, returning `None` on any failure rather than raising:
        applicability is a nice-to-have column, already `None` for an empty
        training set, and two real failure modes must degrade the same way
        rather than take the whole prediction down with them (Task 17 review,
        Important 3 + 4) --
          - a versioned Protocol with no scorecard blob of its own (nothing
            has retrained it yet, so nothing wrote one under its id), and
          - `ScorecardInputs.from_json`'s `cls(**json.loads(...))` round trip
            (Landmine 2) raising `TypeError`/`KeyError` against a future
            schema change, exactly the brittleness that landmine warns against
            deepening -- catching it here is this consumer's forward-compatible
            read path.
        """
        try:
            raw = self._store.get_bytes(scorecard_inputs_key(protocol.workspace_id, protocol.id))
            return ScorecardInputs.from_json(raw).train_structures
        except (FileNotFoundError, TypeError, KeyError, ValueError):
            return None


@dataclass(frozen=True, kw_only=True)
class GetRunQuery:
    run_id: uuid.UUID


class GetRun:
    """Read a Run by id, scoped to the caller's workspace -- the poll endpoint
    every kind of Run (training or prediction) shares."""

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    async def __call__(
        self, query: GetRunQuery, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        run = await self._runs.get(auth.workspace_id, query.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(query.run_id)))
        return Success(run)


@dataclass(frozen=True, kw_only=True)
class CancelRunCommand:
    run_id: uuid.UUID


class CancelRun:
    """Cancel a pending or running Run. `Run.cancel()` owns the actual rule
    (only `pending`/`running` -> `cancelled`; a terminal Run raises
    `ConflictError`) -- this use case's only job is loading the right row and
    persisting the transition."""

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    async def __call__(
        self, command: CancelRunCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(command.run_id)))
        try:
            run.cancel()
        except DomainError as error:
            return Failure(error)
        await self._runs.update(run)
        return Success(run)


@dataclass(frozen=True, kw_only=True)
class PredictedReadout:
    """One Readout's value on one row, carrying the unit and direction the
    Protocol declared for it (Task 17 review, Important 5) -- without these a
    predicted number cannot be lined up against a measurement, which is the
    entire reason `Readout` is derived from the Dataset's TargetSpec in the
    first place. `value` alone would make a chemist re-fetch the Protocol just
    to know what a number means."""

    value: float
    unit: str | None
    direction: str | None


@dataclass(frozen=True, kw_only=True)
class PredictionRow:
    """One scored compound -- everything a triage grid needs to render and
    filter a row without a second call. See this module's docstring for why
    `uncertainty` and `applicability` are shaped the way they are."""

    structure: str
    row_id: int
    readouts: dict[str, PredictedReadout]
    uncertainty: float | None
    applicability: float | None
    # Both None on results written before 2026-10-02; see `RunPrediction`.
    input_row: int | None
    compound_id: str | None


@dataclass(frozen=True, kw_only=True)
class GetPredictionResultsQuery:
    run_id: uuid.UUID
    cursor: str | None = None
    limit: int | None = None
    sort: SortSpec | None = None
    filters: tuple[RangeFilter, ...] = ()


class GetPredictionResults:
    """Read one page of a prediction Run's scored compounds back from Parquet.

    Offset-based, not the keyset cursor `application/pagination.py` uses for
    Protocol/Dataset listings: those paginate a live table that rows are
    concurrently inserted into, where OFFSET silently skips or repeats a row
    that moves mid-scroll. A Run's results file is written once, by the
    worker, and never changes again -- there is nothing for a plain integer
    offset to race, so the simpler mechanism is the honest one here.
    """

    def __init__(
        self, runs: RunRepository, protocols: ProtocolRepository, store: BlobStore
    ) -> None:
        self._runs = runs
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, query: GetPredictionResultsQuery, auth: AuthContext | None = None
    ) -> Result[PageResult[PredictionRow], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, query.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(query.run_id)))
        if run.kind is not RunKind.PREDICTION:
            return Failure(NotFoundError("Prediction results", str(query.run_id)))
        if run.status is not RunStatus.READY:
            detail = run.error_message if run.status is RunStatus.FAILED else None
            return Failure(
                ConflictError(
                    "Results are available only for completed runs; "
                    f"this run is {run.status.label}.",
                    detail=detail,
                )
            )

        protocol_id = uuid.UUID(run.params["protocol_id"])
        protocol = await self._protocols.get(auth.workspace_id, protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(protocol_id)))

        try:
            offset = int(query.cursor) if query.cursor else 0
            if offset < 0:
                raise ValueError("negative offset")
        except ValueError:
            return Failure(
                ValidationError(
                    "Invalid pagination cursor",
                    detail="Pass back the `next_cursor` from the previous page unmodified.",
                )
            )
        limit = clamp_limit(query.limit)

        try:
            raw = self._store.get_bytes(predictions_key(run.workspace_id, run.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Prediction results", str(run.id)))
        # ponytail: reads and holds the entire results Parquet in memory for
        # every page request, not just the page asked for -- fine at today's
        # per-run compound-set sizes. Upgrade path if a run's results grow
        # large: polars' `scan_parquet` (lazy, pushdown-capable) instead of
        # `read_parquet`, or a precomputed row-group index for true partial reads.
        frame = pl.read_parquet(io.BytesIO(raw))

        viewed = apply_result_view(
            frame,
            columns={readout.name for readout in protocol.readouts}
            | {"uncertainty", "applicability"},
            sort=query.sort,
            filters=query.filters,
        )
        if not is_successful(viewed):
            return Failure(viewed.failure())
        frame = viewed.unwrap()

        # Fetch one more row than asked for: its presence is what says there is
        # another page, cheaper and more honest than a second COUNT query.
        page = frame.slice(offset, limit + 1).to_dicts()
        next_cursor = None
        if len(page) > limit:
            page = page[:limit]
            next_cursor = str(offset + limit)

        items = [
            PredictionRow(
                structure=row["structure"],
                row_id=row[ROW_ID],
                readouts={
                    readout.name: PredictedReadout(
                        value=row[readout.name], unit=readout.unit, direction=readout.direction
                    )
                    for readout in protocol.readouts
                },
                uncertainty=row["uncertainty"],
                applicability=row["applicability"],
                # `.get`: results files written before these columns existed stay readable.
                input_row=row.get("input_row"),
                compound_id=row.get("compound_id"),
            )
            for row in page
        ]
        return Success(PageResult(items=items, next_cursor=next_cursor))
