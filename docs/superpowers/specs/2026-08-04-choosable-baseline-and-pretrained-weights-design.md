# Choosable baseline, and pretrained weights for chemprop

**Date:** 2026-08-04
**Status:** approved, not implemented
**Follows:** `docs/superpowers/HANDOFF-chemprop-and-baselines.md` §4.1 and §4.3
**Sibling spec:** `2026-08-04-retry-a-failed-run-design.md` (§4.2, independent)

## Why these are one design and not two

The handoff states them as separate asks: choose the baseline to compare against, and
start chemprop from pretrained weights with "no-pretrained vs pretrained as a baseline
comparison". They collapse into one mechanism.

Today the baseline is fit with defaults and nothing else:

```python
baseline_conditions = validate_conditions(baseline_manifest, {})   # train_protocol.py:361
```

Make the baseline a *(engine, conditions)* pair rather than just an engine, and
"pretrained vs not" stops being a second axis. It is `chemprop-dmpnn{pretrained:
CheMeleon}` measured against `chemprop-dmpnn{pretrained: none}` — an ordinary baseline
choice. The `baseline_is_self` short-circuit already compares engine id *and* conditions
(`train_protocol.py:378-380`), so it distinguishes those two correctly with no change.

Choosing the engine alone would have forced a separate "compare against the untrained
variant" concept, because both sides of that comparison are the same engine id.

## What does not change

A baseline comparison stays **mandatory and never skippable**. The design doc for Phase 1
treats it as the product's thesis and the frontend's verdict copy ("No better than the
baseline") assumes a comparison always exists. Choosable is not skippable.

`EngineManifest.is_baseline` and `EngineRegistry.baseline()` survive unchanged. The flag
now means *the default baseline* — what you get when the request names none. The
"exactly one" invariant in `registry.py:53-62` still holds and still guards the default.

---

## Part 1 — the baseline becomes a choosable pair

### Request and command

`TrainProtocolBody` (`interface/routes/protocols.py:62`) gains two optional fields:

```python
baseline_engine_id: str | None = None
baseline_conditions: dict[str, Any] = Field(default_factory=dict)
```

`model_config` is `extra="forbid"`, so old clients that omit them are unaffected and
typos are rejected.

`TrainProtocol.__call__` (`:267-304`) resolves `baseline_engine_id` **synchronously**,
the same way `engine_id` is resolved at `:271-274`, so an unknown baseline is a 404
before anything is enqueued. Conditions stay unvalidated until the worker, also matching
today's split (`:237-249`).

The **resolved** id goes into `run.params`, never `None`. `params` is write-once and the
run must be self-describing: if a queued run stored `None` and the registry's default
baseline changed before a worker picked it up, the run would silently measure against a
different model than the one the user was shown. `TrainProtocolCommand.from_params` reads
both new keys with `.get()` defaults so runs written before this change still parse.

### Cache key

`compute_cache_key` at `:293-298` gains `baseline_engine_id` and
`baseline_conditions=sorted(...)`. Training runs are not cache-looked-up today — only
predictions reuse a cached Run — but the key is computed and stored, and fit-result
caching is a live deferred item (handoff §5). Including the baseline now is two lines and
stops a future cache from serving a run whose baseline was different.

### RunTraining

`train_protocol.py:359-361` becomes a resolution from the command rather than from the
registry:

```python
baseline = (
    self._engines.get(command.baseline_engine_id)
    if command.baseline_engine_id
    else self._engines.baseline()
)
baseline_manifest = baseline.manifest()
baseline_conditions = validate_conditions(baseline_manifest, command.baseline_conditions)
```

Three consequences, each needing a fix.

**The baseline needs its own task-capability check.** The guard at `:351-356` covers only
the chosen engine. That was safe while the baseline was always `ecfp4-randomforest`,
which declares both tasks. A chosen baseline may declare only one. Same `ValidationError`
message shape, applied to `baseline_manifest`, and placed before any compute so a
mismatch costs nothing.

**Lane routing becomes ambiguous.** The run is enqueued on the chosen engine's lane
(`:304`). Choose `ecfp4-randomforest` (lane `default`) with a `chemprop-dmpnn` baseline
(lane `gpu`) and the run lands on the default-lane worker, which fits the chosen engine
fine and then dies in `_require_chemprop()`. Route on whichever side needs a non-default
lane:

```python
# ponytail: two lanes, so "the non-default one" is unambiguous. A run wanting two
# *different* non-default lanes has no home; split it into two runs if a third lane
# ever exists.
lane = next(
    (m.lane for m in (manifest, baseline_manifest) if m.lane != DEFAULT_LANE),
    DEFAULT_LANE,
)
```

This is resolved in `TrainProtocol` at enqueue time, where both manifests are already in
hand.

**`baseline_conditions` must reach the page.** Without it, a pretrained-vs-not run renders
as `chemprop-dmpnn` versus `chemprop-dmpnn` — a comparison the page cannot explain. It is
a four-link chain, and all four links are required or the field dead-ends:

1. `ScorecardInputs` (`train_protocol.py:121-206`) — the written blob. Takes
   `field(default_factory=dict)`, because `from_json` is `cls(**json.loads(data))`
   (`:206`) and a field without a default would break reads of every blob written before
   today. The dataclass is `kw_only`, so ordering is free.
2. The `Scorecard` domain dataclass (`domain/execution/scorecard.py`).
3. `build_scorecard` (`application/execution/build_scorecard.py`), which renders 2 from 1.
4. `ScorecardResponse` (`interface/routes/protocols.py:170-229`) — note this class already
   renames two fields on the way out (`target_unit`→`unit`, `target_direction`→
   `direction`), so it is a real mapping and not a passthrough.

### Frontend

`train-protocol-form.tsx` gains a **Compare against** select below the engine select,
listing the same `enginesForTargetKind(...)` set, defaulted to the manifest with
`is_baseline: true`, plus a collapsed **Baseline settings** block reusing
`<ConditionFields>` unmodified against the baseline manifest.

Two pieces of copy become false and must go dynamic:

- `train-protocol-form.tsx:174-179` — "This engine is the baseline, so there is nothing
  to compare it against." No longer true merely because the chosen engine is flagged; you
  can now give it a different baseline. It should fire when the two sides genuinely match
  — same engine id *and* deep-equal conditions — which is exactly the `baseline_is_self`
  condition, evaluated client-side.
- `scorecard-view.tsx:122-127` — the `is-baseline` verdict body hardcodes "You trained
  ECFP4 + RandomForest". It must name `scorecard.engine_id`.

`scorecard-view.tsx:160-164` renders `baseline_engine_id` in prose. When the baseline
engine equals the chosen engine, that sentence must also name the differing conditions,
or the pretrained comparison reads as a model against itself.

`openapi.json` is regenerated and `make generate-api` re-run; the generated
`TrainProtocolBody`, `ScorecardResponse` and `EngineManifestResponse` types come along.

---

## Part 2 — pretrained weights for chemprop

### The condition

One enum on `chemprop-dmpnn`'s manifest:

```python
ConditionSpec(
    key="pretrained",
    label="Pretrained weights",
    type=ConditionType.ENUM,
    default="none",
    options=("none", "CheMeleon"),
    help="Start from a foundation model's learned representation instead of random "
         "weights. CheMeleon was pretrained on ~1M PubChem molecules against classical "
         "descriptors, and fixes the message-passing size and depth.",
)
```

Plain data, so `test_engine_contract.py:139-161`'s JSON round-trip tripwire passes
untouched, and `condition-fields.tsx:56-71` already renders `enum` as a `<Select>` — the
picker costs nothing. The option string *is* the display label; `ConditionSpec` has no
per-option label mechanism and does not need one for two values.

Adding a second weight set later is one tuple entry plus one registry row.

### What CheMeleon actually is

Verified against the installed chemprop 2.3.0 wheel and Zenodo, not recalled:

| | |
|---|---|
| Artifact | `https://zenodo.org/records/15460715/files/chemeleon_mp.pt` |
| Size / MD5 | 34,859,448 bytes · `6a80b54fdb7de37ef0374d302f01e8ce` |
| Licence | MIT (weights and code both) |
| Contents | Two keys only: `hyper_parameters`, `state_dict`. A message-passing block, no predictor head. |
| Paper | arXiv:2506.15792, Burns et al. |

It is **not** loadable through `MPNN.load_from_checkpoint` — that raises `KeyError:
'metrics'`, because the file is not a Lightning checkpoint. The handoff's guess that
`load_from_checkpoint` would be the entry point is wrong, and this line is here so nobody
re-derives it.

### Loading it

The documented 2.3.0 Python path, which replaces the `BondMessagePassing(d_h=hidden,
depth=depth)` construction at `chemprop_dmpnn.py:230-235`:

```python
checkpoint = torch.load(_weights_path("CheMeleon"), weights_only=True)
message_passing = BondMessagePassing(**checkpoint["hyper_parameters"])
message_passing.load_state_dict(checkpoint["state_dict"])
```

Two knock-on changes in the same block, both load-bearing:

- The FFN's `input_dim` must be `message_passing.output_dim` (2048), **not** the
  `message_hidden_dim` condition. The existing comment at `:226-227` says `input_dim`
  must equal the message-passing `d_h` or the first layer is built for the wrong width —
  that stays true, but under pretraining `d_h` comes from the checkpoint.
- `batch_norm` flips from `True` to `False`, matching chemprop's own
  `chemeleon_foundation_finetuning` notebook.

Aggregation is already `MeanAggregation()`, which is what CheMeleon requires. The V2 atom
featurizer is already the default. Nothing else in `train()` moves, and `predict()` is
untouched — the artifact is still a Lightning `.ckpt` written by `trainer.save_checkpoint`,
so `MPNN.load_from_checkpoint` at `:298` still reconstructs the architecture from the
checkpoint's own saved hyperparameters, pretrained or not.

### Where the file lives

A new setting beside the blob options in `settings.py`:

```python
pretrained_weights_dir: str = "~/.cache/daikon-studio/weights"
```

On first use the engine downloads to a temp file in that directory, checks the MD5, then
`os.replace`s it into place. The atomic rename is what keeps two workers on one box from
reading a half-written file. The MD5 check is not ceremony: a truncated download
otherwise surfaces as an unreadable-tensor error with no hint that the network was the
cause, and this is a trust boundary — bytes from the internet becoming model weights.

`Dockerfile.gpu` bakes the file in at build time so a production worker never reaches
Zenodo, and so an air-gapped deployment works. The download path stays for local dev.

### The honesty problem

CheMeleon pins `depth=6` and `d_h=2048`. Two conditions the user can set become inert.

Silently ignoring them is the exact dishonesty this codebase refuses everywhere else —
the whole Scorecard exists to stop a model looking better than it is, and a stored
`depth: 3` that the fit never used is a small lie of the same kind. Rejecting them does
not work either: the form posts every condition it renders, seeded from defaults, so a
reject-if-supplied rule would reject every request.

So the **form** sets them. When `pretrained` is not `none`, `depth` and
`message_hidden_dim` are set to the pinned values and their inputs disabled, with a
caption naming CheMeleon as the reason. Both pinned values sit inside their declared
bounds — `depth` max is 6, `message_hidden_dim` max is 2400 — so the stored conditions
stay truthful with no new mechanism, no validation change, and no `ponytail:` debt.

The engine still reads the checkpoint's own hyperparameters rather than trusting those
condition values, because a request that bypasses the form must not be able to build a
mismatched network.

---

## Testing

| Level | What it locks |
|---|---|
| Unit, `application/engines` | Baseline resolution: named id wins, absent id falls back to `registry.baseline()`, unknown id is a 404 before enqueue. |
| Unit, `execution` | The lane union — chosen `default` + baseline `gpu` enqueues on `gpu`. Extends `tests/unit/execution/test_lanes.py`. |
| Unit, `execution` | A baseline that cannot serve the dataset's task raises `ValidationError` before any fit. |
| Unit, contract | Already automatic: `test_every_registered_manifest_round_trips_through_json` covers the new enum condition with no new test. |
| Unit, `ScorecardInputs` | `from_json` reads a blob written without `baseline_conditions` — the regression this default exists to prevent. |
| Integration | A training run with an explicit baseline lands that engine id and those conditions in the written `ScorecardInputs`. |
| Unit, chemprop | CheMeleon builds a network whose `output_dim` is 2048 and whose FFN accepts it. Gated by `pytest.importorskip("chemprop")` **and** a skip unless the weights are already cached, so CI never downloads 35 MB. |
| Frontend | `verdict.ts` unit tests extend to the case where the baseline is the same engine with different conditions. |

The real end-to-end check is a BBBP run of `chemprop-dmpnn{pretrained: CheMeleon}` against
`chemprop-dmpnn{pretrained: none}`, on the gpu lane, producing a Scorecard whose verdict
band names both sides distinctly. `OMP_NUM_THREADS=1` remains load-bearing for it
(handoff §3).

## Accepted risks

- **A second CheMeleon-shaped weight set may not fit the enum.** A v1-format checkpoint
  needs `chemprop convert` and the V1 featurizer, which the condition cannot express. The
  registry row is where that would go; not built, because there is one weight set.
- **Uploaded checkpoints are out of scope**, per the decision to ship preconfigured sets
  only. `StoreUpload`/`upload_key` in `application/data/create_dataset.py` remains the
  obvious home if that changes.
- **A pretrained fit is slower to start** by one 35 MB download on a cold local cache.
  Production never pays it.
