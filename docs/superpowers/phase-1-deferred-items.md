# Phase 1 — deferred items

Every item consciously deferred during the Phase 1 backend build, with the reasoning
recorded at the time. Triaged by the final whole-branch review: none block merge.
Kept because the reasoning is the valuable part — each entry says why it was safe to
defer, which is what a future reader needs in order to decide it no longer is.

Items marked CARRY INTO FINAL were escalated and resolved; they remain here for history.

- Task 1: minor (deferred): Makefile:34 `pg_isready -U daikonstudio` vs actual POSTGRES_USER
- Task 2: minor (deferred): shared primitives test covers 5 behaviours; PageResult, Provenance/Citation
- Task 3: minor (deferred): to_thread hop for command.upgrade could be split into sync-migrate +
- Task 3: minor (deferred, for Task 20): `make up`/`make migrate` fail at the .env-sourcing step —
- Task 3: minor (deferred, CARRY INTO TASK 11): no persisted regression test for the
- Task 4: minor (deferred): (4) _core.py docstring claims import-safety the guard doesn't provide —
- Task 4: minor (deferred, CARRY INTO TASK 11 — LANDMINE): conftest.py's global fake Sentinel creds
- Task 4: minor (deferred): _core.py's module-level try/except binds _get_request_auth exactly once at
- Task 4: minor (deferred, CARRY INTO TASK 11): test_auth_dependency.py uses importlib.reload on a
- Task 5: PARKED (plan-mandated, accepted risk): FsspecBlobStore uses fsspec's private _fs._parent().
- Task 5: minor (deferred): blob tests verify only through the store's own methods — an in-memory dict
- Task 6: minor (deferred): nearest_neighbour_tanimoto returns float64 on the empty-reference path
- Task 7: minor (deferred): no test for a required condition with no default that is not supplied
- Task 7: minor (deferred): _coerce error messages read "must be a integer"/"must be a enum" (article
- Task 8: minor (deferred): engine identity duck-typed via hasattr(model,"estimators_") rather than an
- Task 8: minor (deferred): joblib remains a declared direct dependency though no first-party code
- Task 9: minor (deferred): prepare_frame silently narrows extra uploaded columns to the first row's
- Task 9: minor (deferred, CARRY INTO TASK 11 — LANDMINE): the PRE-EXISTING valid_rows==0 early return
- Task 9: minor (deferred): the new empty-frame test asserts only height/total_rows/valid_rows, not
- Task 10: minor (deferred): _partition_sizes docstring claims test absorbs rounding error but test can
- Task 10: minor (deferred): unreachable "(acyclic/invalid)" fallback branch in the error message;
- Task 11: minor (deferred): upload size check runs after the body is already spooled to disk;
- Task 11: minor (deferred): persistence models.py:33 carries the same "same hash means same data"
- Important 3 (PARKED, see ruling below): derive_readouts placed in application/catalog/ rather than
- Task 12: PARKED with ruling — derive_readouts placement. Reviewer confirmed it is LEGAL under both
- Task 12: minor (deferred): derive_readouts has no explicit else/raise for an unrecognised TaskType
- Task 12: minor (deferred): repository.py:138 uses a bare `assert isinstance(result, CursorResult)`
- Task 13: minor (deferred, CARRY INTO TASK 17): find_by_cache_key returns the most recent row for a
- Task 13: minor (deferred): a Run stuck RUNNING forever if a worker crashes mid-job — a redelivered
- Task 14: minor (deferred): baseline_is_self skips the 0.66 progress write so progress jumps
- Task 14: minor (deferred): a Protocol-insert failure now orphans TWO blobs (artifact + scorecard)
- Task 15: minor (deferred): metrics types widened to dict[str, float | None] correctly but not
- Task 16: minor (deferred): ArqEnqueuer.aclose() never called from the app lifespan; malformed
- Task 16: minor (deferred, RELEVANT TO TASK 17): ScorecardInputs is a frozen dataclass with no field
- Task 17: minor (deferred): zero ponytail comments in 613 new source lines (whole-Parquet-per-page,
- Task 17: CARRY INTO FINAL REVIEW: GetScorecard.__call__ (application/catalog/get_scorecard.py:65)
- Task 17: minor (deferred): FsspecBlobStore's new URI passthrough assumes BLOB_BASE_URL never changes
- Task 18: minor (deferred, CARRY INTO FINAL REVIEW): EngineManifest.tasks uses TaskType values
- Task 18: minor (deferred): ConditionResponse.type and tasks are plain `str` rather than Literal/enum,
- Task 19: minor (deferred): export omits the lineage the Collection already stores (provenance.note
- Task 19: CARRY INTO FINAL REVIEW (pre-existing, codebase-wide): a `name` over 256 chars returns 500
- Task 19: CARRY INTO FINAL REVIEW (pre-existing, upstream of export): a readout named "uncertainty",
- Task 10: minor (deferred): docstring says reordering "cannot" change the split for a fixed seed;
- Task 20: minor (deferred): the new noise_floor coverage-boundary comment cites test_train_protocol.py,
