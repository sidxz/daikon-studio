# daikon-studio — Phase 1 Design

**Date:** 2026-07-25
**Status:** Proposed
**Scope:** Phase 1 only. Phases 2–5 are sketched at the end for sequencing context, not specified here.

## 1. What this is

daikon-studio is the *in-silico* wing of the DAIKON suite. chem-cellar holds your in-vitro
protocols and the runs that execute them; studio holds your in-silico protocols and the runs
that execute them. Same grammar, different medium.

It is a curator's platform. A scientist pulls data, trains a model, publishes it as a runnable
protocol, runs protocols across compound sets, triages the results, and proposes the curated
output back to chem-cellar, prot-cellar, or daikon-gen3. The deliverable is a curated set of
compounds someone is willing to order. The model is the means.

Studio is its own product with its own UI, in the sister-app constellation (chem-cellar,
prot-cellar, daikon-gen3, docu-store) sharing one Sentinel realm.

### Why the phasing looks like this

Phase 1 walks the *entire* loop using the cheapest possible engine — ECFP4 fingerprints with
XGBoost and RandomForest — which is CPU-only, trains in seconds, and needs no GPU queue, no
container isolation, and no scheduler. It is also the baseline required for scientific honesty
regardless of what else ships.

The point of Phase 1 is not the model. It is to exercise every noun in the terminology below
end to end, so the domain model is proven before any expensive infrastructure is built on top
of it. If a noun is wrong, we find out in Phase 1 rather than Phase 4.

## 2. Terminology

This vocabulary is load-bearing and the most likely thing to erode. It belongs in the spec, not
in folklore.

### Core objects

| Term | Type name | What it is | chem-cellar twin |
|---|---|---|---|
| **Engine** | `Engine` | Packaged capability — manifest + typed entrypoints. Trains and/or predicts. | (the instrument/technique) |
| **Protocol** | `InSilicoProtocol` | A trained, configured, publishable artifact. What other people run. | `Protocol` |
| **Readout** | `Readout` | One declared output: name, type, unit, direction, description. | `ReadoutDefinition` |
| **Condition** | `Condition` | One declared knob: default, bounds, choices. Overridable per run. | `ConditionDefinition` |
| **Dataset** | `Dataset` | Frozen snapshot — source query + filters + split, content-hashed. | frozen `Collection` |
| **Collection** | `Collection` | A named set of compounds. Output-only in Phase 1. | `Collection` |
| **Catalog** | *(view)* | Engines and Protocols promoted to the global workspace. | — |

A Protocol versions the way a chem-cellar Protocol does — `parent_protocol_id` self-FK plus a
version number, locked on publish — rather than getting a separate version entity. Readouts and
Conditions live on the Protocol, which is what makes a published Protocol self-describing enough
for the UI to render its run form without knowing anything about the Engine underneath.

### Execution

| Term | Type name | What it is | chem-cellar twin |
|---|---|---|---|
| **Run** | `Run` | One execution. Kinds: `training`, `prediction`, `generation`. | `Run` / `ImportRun` |
| **Sweep** | `Sweep` | N training runs over one Dataset, varying Engine or hyperparameters. | — |
| **Scorecard** | `Scorecard` | The eval view on a training run. | `qc_metrics` |
| **Triage** | *(stage)* | Filtering a run's results down to a Collection. | hit criteria + flags |
| **Proposal** | `Proposal` | A curated Collection offered to another app for a human to accept. | — |

Run carries the established job shape: `pending/running/ready/failed/cancelled`, a `progress`
column polled at 2s, and a content-addressed cache key so identical inputs return the cached
result instead of recomputing.

Sweeps are only meaningful because Datasets are immutable and carry their own split — that is
what guarantees the N results are comparable. Triage is deliberately a stage rather than an
aggregate; the artifact you save is a Collection.

### Data and provenance

| Term | Type name | What it is | Source |
|---|---|---|---|
| **Source** | `Source` | Pluggable data adapter: CSV, chem-cellar, prot-cellar. | gen3 `DataSource` |
| **Split** | *(field on Dataset)* | random / scaffold / Butina / UMAP-cluster / temporal / by-batch. | — |
| **Artifact** | `Artifact` | Blob on object storage — weights, Parquet snapshots, plots. | docu-store `BlobStore` |
| **Prediction** | `Prediction` | One value for one compound, with uncertainty and applicability flag. | — |
| **Provenance** | `Provenance` | VO with `generation_method=ai_predicted`. | prot-cellar, verbatim |
| **Lineage** | *(view)* | The DAG over everything above. | — |

Split is a field rather than an entity, but it is a *named, visible* field — never a silent
80/10/10. Provenance is copied wholesale from prot-cellar rather than reinvented, which is what
lets a prediction flowing back into chem-cellar render as AI-generated and flip to manual the
moment a human edits it.

### Words we deliberately do not use

| Word | Why not |
|---|---|
| **Model** | Ambiguous between architecture and trained artifact. Split into Engine + Protocol. |
| **Method** | Near-synonym of Protocol in this domain; collides with Python methods. |
| **Recipe** | Considered and dropped — Protocol carries that meaning. |
| **Experiment** | Vague. Run and Sweep cover it precisely. |
| **Pipeline** | Implies a DAG editor we are not building in v1. |
| **Assay** | Belongs to chem-cellar and means the wet measurement. |
| **Screen** | Taken by daikon-gen3's pipeline stage. |
| **Job** | Internal arq term only. Users always see Run. |

There is no `models` table and no `Model` class. People will say "model" colloquially forever and
that is fine, but the architecture depends on Engine and Protocol staying distinct.

## 3. Phase 1 scope

### In

- CSV upload as the only Source.
- Dataset creation: target column selection, structure validation, deduplication, split
  (random or scaffold), frozen Parquet snapshot with a content hash.
- Two in-tree Engines: `ecfp4-xgboost` and `ecfp4-randomforest`. Regression and binary
  classification.
- Training a Protocol from (Dataset × Engine × conditions), producing readouts derived from the
  Dataset's target spec.
- Scorecard with a mandatory baseline comparison, an optimism gap, an assay-noise floor where
  duplicates existed, worst-20 residuals as rendered structures, and applicability-domain
  coverage.
- Publishing a Protocol (locks it, makes it runnable by others in the workspace).
- Prediction runs: a published Protocol against an uploaded CSV.
- Triage grid over run results; save a selection as a Collection; export CSV/SDF.
- Sentinel auth in authz mode, workspace-scoped tenancy.
- Lineage *captured* on every entity.

### Out (deferred, with the phase that owns each)

| Deferred | Phase |
|---|---|
| GPU engines, container isolation, image builds, job scheduling | 2 |
| chemprop, cage_fusion as registered engines | 2 |
| Sweep UI, ensembles, additional splits (Butina/UMAP/temporal) | 2 |
| Lineage DAG visualization (xyflow canvas) | 2 |
| chem-cellar / prot-cellar as Sources | 3 |
| Proposal write-back, cross-workspace publishing, global Catalog | 3 |
| Generation runs and their triage (novelty, synthesizability) | 4 |
| Third-party Engine registration, conformance suite, sandboxing | 5 |

Also explicitly not in Phase 1 and not currently planned: Temporal, Kubernetes, MLflow, Weights
& Biases, a model hub. The Run table with a metrics column plus artifacts on blob storage is
experiment tracking at this scale, and MLflow's real value is a comparison UI we would rebuild
anyway to make it legible to a biochemist.

## 4. Architecture

### Stack

Copied from prot-cellar, the lighter of the two siblings.

- **Backend:** Python 3.13, FastAPI, SQLAlchemy 2.0 async + asyncpg, Alembic, Pydantic v2,
  Lagom (DI), `returns` (Result), structlog, PyJWT, httpx, `sentinel-auth-sdk`.
- **Jobs:** arq over Valkey. Not Temporal — prot-cellar already made this call and documented
  the ceiling; we inherit both the call and the upgrade path.
- **DB:** plain PostgreSQL 16. No RDKit cartridge: Phase 1 needs in-process RDKit for
  featurization and scaffolds, not SQL substructure search.
- **Storage:** `FsspecBlobStore` port copied from docu-store, `BLOB_BASE_URL` pointed at the
  existing MinIO or the NFS share. Not an HTTP hop through docu-store, which is a document
  service and has no blob endpoint.
- **Science:** rdkit, scikit-learn, xgboost, polars.
- **Frontend:** Next.js 16, React 19, TS, Tailwind v4, radix/shadcn locally,
  `@structflo/daikon-design-tokens`, TanStack Query v5, Zustand, AG Grid, `@rdkit/rdkit` WASM
  for structure rendering, orval against a committed `openapi.json` snapshot.
- **Ports:** backend 8002, frontend 3002, Postgres 5434, Valkey 6381 — offset from cellar
  (8000/3000/5432/6379) and prot-cellar (8001/3001/5433/6380) so all three run side by side.

Polars rather than pandas for dataset materialization: it is substantially faster on the
snapshot path and writes Parquet natively, which is the storage format for a frozen Dataset.

### Bounded contexts

Three, not five. Import-linter enforces the same contracts as the siblings: layer order
`interface > infrastructure > application > domain`, domain purity, and context independence
with cross-context references as bare indexed UUIDs.

| Context | Aggregates |
|---|---|
| `catalog` | `Engine`, `InSilicoProtocol` (owns `Readout`, `Condition`) |
| `data` | `Dataset` (owns `SplitSpec`, `TargetSpec`, `ValidationReport`), `Collection` |
| `execution` | `Run` (owns `Scorecard` for training runs) |

Engine and Protocol share a context because they share a lifecycle in the UI — they are both
things you browse and pick from — and because a Protocol references its Engine directly rather
than by bare UUID.

### The Engine contract

Phase 1 implements the contract as an in-tree manifest plus registry, directly mirroring
prot-cellar's `PluginManifest` / `IngestionPlugin` / `Sink` pattern. No containers, no dynamic
loading, no image builds — those arrive in Phase 5 when third parties actually need them, and
the manifest is deliberately shaped so the envelope serializes to HTTP unchanged when it moves
out of process.

```python
# application/engines/manifest.py

@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    type: ConditionType          # STRING | INTEGER | NUMBER | ENUM | BOOL
    required: bool = False
    default: object | None = None
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()
    help: str | None = None

@dataclass(frozen=True)
class EngineManifest:
    id: str                      # "ecfp4-xgboost"
    version: str                 # "1.0.0"
    name: str
    description: str
    tasks: tuple[TaskType, ...]  # REGRESSION | BINARY_CLASSIFICATION
    conditions: tuple[ConditionSpec, ...]
    is_baseline: bool = False
```

`ConditionSpec` is constrained exactly as far as prot-cellar's `ParamField` is — enough that the
frontend renders the form directly without a JSON-schema-form library, and no further.

```python
# application/engines/protocol.py

class Engine(Protocol):
    @staticmethod
    def manifest() -> EngineManifest: ...
    def train(self, ctx: TrainContext) -> TrainResult: ...
    def predict(self, ctx: PredictContext) -> PredictionFrame: ...
```

Structural typing, no base class — same as prot-cellar's plugin protocol.

**Engine methods are synchronous.** Training and inference are CPU-bound and would block the
event loop; the worker calls them through `asyncio.to_thread`. Engine authors write plain sync
code and never think about async. This is a deliberate divergence from prot-cellar's async
`run()`, and the reason belongs in the module docstring.

The framework — not the engine — attaches provenance, `workspace_id`, the source run id, engine
id and version, and reports progress. Engine authors write featurize-fit-predict and nothing
else.

### Readouts are derived, not declared by the Engine

An Engine cannot declare its concrete outputs, because they depend on what you trained it on.
The chain is:

1. `Dataset.target: TargetSpec{column, kind, unit?, direction?}` — the scientist states what
   they are predicting and in what units when they create the Dataset.
2. Engine declares only which `TaskType`s it supports.
3. At training time the Protocol materializes concrete `Readout`s from the TargetSpec and the
   task.

This is the mechanism by which a predicted IC50 arrives in the same unit and with the same
direction as a measured one. It is the single most important structural detail in the design and
the thing that makes predictions comparable to experiments downstream.

### Storage layout

```
{workspace_id}/datasets/{dataset_id}/snapshot.parquet
{workspace_id}/protocols/{protocol_id}/artifact/*
{workspace_id}/runs/{run_id}/results.parquet
```

Content addressing follows chem-cellar's `UmapJob` pattern. `Dataset.content_hash` covers the
snapshot bytes, filters, split strategy and seed. A Run's cache key is
`hash(protocol_id, protocol_version, input_hash, conditions)`; a hit returns the cached result
rather than recomputing.

### Data flow

```
CSV upload
  → validate + canonicalize + dedupe        (rejects at the door)
  → Dataset (frozen Parquet + content hash + split)
  → TrainingRun × {chosen engine, baseline engine}
  → InSilicoProtocol (readouts derived from TargetSpec)  + Scorecard
  → publish (locks the protocol)
  → PredictionRun (protocol × uploaded CSV)
  → triage grid
  → Collection
  → export
```

Every arrow records what produced what. The lineage is captured in Phase 1 and visualized in
Phase 2 — capturing it later is impossible, visualizing it later is trivial.

## 5. The honesty layer

This is not a nice-to-have and it is not deferrable. Training is easy; the failure mode of every
no-code ML product surveyed is a model that looks good on a random split and fails
prospectively. The differentiator is making that visible by default.

### Validation at upload, refusing at the door

On Dataset creation, parse the structure column and report before anything is frozen:

- invalid structures, with row numbers — rejected, not silently dropped
- duplicates after canonicalization — **collapsed, with the count reported**
- salts and mixtures, stereochemistry loss — flagged

Collapsing duplicates depends on the target kind. For a numeric target, replicates are averaged
and the spread is retained. For a binary target, agreeing replicates collapse silently and
**conflicting replicates are rejected with their row numbers** rather than resolved by majority
vote — a compound labelled both active and inactive is a data problem the scientist must decide
about, not one we should paper over.

Deduplicating before splitting eliminates train/test structure leakage by construction rather
than warning about it afterwards.

The within-duplicate spread on numeric targets is kept because it is a free estimate of assay
noise, and assay noise is the honest floor for model error. It appears on the Scorecard. Binary
targets have no equivalent, so that Scorecard row is simply absent for them.

### Scorecard

For every training run, all on the same Dataset and therefore the same split:

- the primary metric, leading with MCC and balanced accuracy for classification (never bare
  accuracy — a 99.9%-negative dataset yields a 99.9%-accurate useless model)
- **the mandatory baseline** — the same metric for ECFP4+RandomForest. Not optional, not a
  checkbox. In the Polaris ADMET competition, fingerprint baselines placed around 20th of 66
  teams; a user whose engine cannot beat one needs to know on the first screen
- **the optimism gap** — where the split is scaffold, the same engine's random-split number
  alongside it, so the difference is visible rather than inferred
- **the assay-noise floor**, where duplicates existed
- **worst-20 residuals as rendered structures, grouped by Murcko scaffold**, so a chemist reads
  "it fails on the sulfonamides" instead of "R² = 0.61"
- **applicability domain** — nearest-neighbour Tanimoto from each test compound to the training
  set, as a distribution and as a per-compound flag

`ponytail:` training three protocols per request (engine, baseline, engine-on-random-split) is
free for CPU engines and will not be for GPU engines. When Phase 2 lands, the random-split
comparison becomes opt-out for expensive engines; the baseline stays mandatory.

### Triage

The triage grid is where the honesty layer pays off, and it is the primary artifact of a
prediction run — not a metrics dashboard. Sortable and filterable, one rendered structure per
row, predicted value with uncertainty, applicability flag, multi-property filters, then save
selection as a Collection.

A scientist makes better decisions from "7 of your top 20 are outside the domain of
applicability" than from any aggregate metric.

## 6. API surface

REST, `/api/v1/<resource>`, one `APIRouter` per file, cursor pagination via `PaginatedResponse[T]`,
`result_to_response` mapping `Result[T, DomainError]` onto the shared status map. All copied from
the siblings.

```
GET   /api/v1/engines                          → list[EngineManifestResponse]

POST  /api/v1/datasets/uploads     multipart   → {upload_ref}
POST  /api/v1/datasets             201         → Dataset + ValidationReport
GET   /api/v1/datasets                         → PaginatedResponse[DatasetResponse]
GET   /api/v1/datasets/{id}

POST  /api/v1/protocols            202         → training Run
GET   /api/v1/protocols                        → PaginatedResponse[ProtocolResponse]
GET   /api/v1/protocols/{id}
GET   /api/v1/protocols/{id}/scorecard
POST  /api/v1/protocols/{id}/publish  204

POST  /api/v1/runs                 202         → prediction Run
GET   /api/v1/runs/{id}                        → status, progress, phase
GET   /api/v1/runs/{id}/results                → PaginatedResponse[PredictionResponse]
POST  /api/v1/runs/{id}/cancel        204

POST  /api/v1/collections          201         → Collection (from a triage selection)
GET   /api/v1/collections/{id}/export?format=csv|sdf
```

202 for enqueued work, client polls `GET /{id}` at 2s with `refetchInterval` auto-stopping on a
terminal status — prot-cellar's `use-imports.ts` pattern verbatim.

## 7. Auth and tenancy

Sentinel in authz mode, in the same realm as chem-cellar, prot-cellar, daikon-gen3 and
docu-store, so sign-in and workspace context are shared. `sentinel.protect()` registered before
CORSMiddleware — Starlette middleware is LIFO and the reverse order ships 401s without CORS
headers.

Service actions declared at startup, best-effort (never fatal — JWKS fetch stays fatal, action
registration does not):

```
studio:read  studio:write  studio:train  studio:publish  studio:admin_config
```

`studio:register_engine` arrives in Phase 5 with third-party engines, since registering an
Engine means running third-party code on our hardware and warrants its own admin-gated action
plus a review step.

Tenancy is `workspace_id` on every row, taken from `auth.workspace_id` and never from a body or
URL. Guards are the first two lines of every use case. `GLOBAL_WORKSPACE_ID` is reserved now for
the Phase 3 public Catalog but unused in Phase 1.

## 8. Error handling

Inherited unchanged from the siblings: `Result[T, DomainError]` throughout, never raising for
expected failures; the shared `DomainError` → status map (NotFound 404, Validation 422, Conflict
409, Authorization 403, DataLocked 423); optimistic concurrency via `version` with a 409 and
`retry: true`.

Two studio-specific cases:

- **Dataset validation failure** is a 422 carrying the full `ValidationReport`, not a bare
  message. The report is the useful part.
- **Engine failure inside a run** records `status=failed` with the exception on the Run row and
  re-raises so arq logs it — prot-cellar's `run_import` shape. A failed run is never billed,
  never silently retried, and always cancellable.

## 9. Testing

`tests/{unit,integration,api,fakes,factories}`, testcontainers for Postgres, `FakeAuth` in
fakes, `dependency_overrides` on the stable `get_auth` wrapper.

The Phase 1 test that matters most is an end-to-end one: upload a real CSV from
`workspace/lab-ai/pains/dataset/`, create a Dataset, train both engines, publish, run against a
held-out CSV, triage, export — asserting on the Scorecard's baseline comparison rather than on
absolute metric values.

## 10. Success criteria

Phase 1 is done when:

1. A scientist uploads a CSV and publishes a working Protocol in under two minutes without
   reading documentation.
2. A second user in the same workspace runs that Protocol against their own CSV and exports a
   triaged Collection.
3. The Scorecard makes it immediately obvious when a Protocol is no better than the baseline.
4. Every noun in §2 has been exercised end to end by a real user flow.

Criterion 4 is the actual goal. The others are how we verify it.

## 11. Open questions and risks

- **Engine confirmed, but the Layer-1 name is the one decision that propagates everywhere.**
  Recorded here as `Engine`; changing it after Phase 1 means touching every table name and route.
- **`Sweep` is defined in §2 but only partially exercised in Phase 1** — the mandatory-baseline
  mechanism is a two-run sweep, but there is no Sweep UI or Sweep aggregate until Phase 2. It is
  listed in the terminology now so the word is reserved and unambiguous.
- **Publishing across workspaces is a data-leakage surface.** A model trained on one
  organisation's proprietary assay data can leak that data through its predictions. Phase 1
  keeps everything workspace-scoped, which defers the question; Phase 3 must answer it before the
  global Catalog opens, with explicit promotion and a review gate.
- **Immutability is what makes a Protocol citable** — dataset hash, split definition, metrics and
  applicability domain, never mutated. This is cheap to get right in Phase 1 and effectively
  unrecoverable if we do not.
- **cage_fusion's `refactor4` branch does not currently boot** — it dropped `cage_fusion/api/`
  and `configs.py` while `entrypoint.sh` still references both, and the FastAPI layer exists only
  on `origin/main`. This blocks Phase 2, not Phase 1, but it should be reconciled before Phase 2
  planning rather than discovered during it.

## 12. Phase sketch beyond 1

Recorded for sequencing only; each gets its own spec.

- **Phase 2 — GPU and comparison.** Container-isolated engines, the job queue with GPU-slot
  semantics, chemprop and cage_fusion as registered engines, Sweep UI, ensembles, remaining
  splits, the lineage canvas.
- **Phase 3 — the suite.** chem-cellar and prot-cellar as Sources, Proposal write-back as
  propose-not-write per daikon-gen3's locked read-and-propose decision, cross-workspace
  publishing and the global Catalog.
- **Phase 4 — generation.** Generation runs, novelty and synthesizability triage, the
  registration path for new structures.
- **Phase 5 — third parties.** Custom image builds, an executable conformance suite, sandbox
  hardening, `studio:register_engine`.
