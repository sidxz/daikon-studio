# Self-hosted runners: design

**Date:** 2026-08-04
**Status:** Approved pending user review

## Problem

Training and prediction jobs currently run on arq workers that need direct
access to Valkey, Postgres, and the blob store. That confines compute to
machines we fully trust with infrastructure credentials. We want GitHub-runner
UX: a user creates a runner in the UI, pastes one command on any machine
(their own GPU box or an external collaborator's), and that machine starts
serving jobs. Down the line, work becomes multi-step (hyperparameter sweeps,
ensembles, featurize→train→evaluate pipelines) across many users, models, and
nodes.

## Decisions taken during brainstorming

1. **Pull, not push.** Runners dial out to the studio; the UI never stores a
   runner address. Works behind NAT, no inbound ports, matches how GitHub
   runners actually work.
2. **Runners live on arbitrary internet machines**, including hardware owned
   by external collaborators. Runners therefore never receive DB/queue/blob
   credentials — everything is mediated by an HTTPS protocol.
3. **One dispatch mechanism.** The HTTPS runner protocol replaces arq for
   local workers too. arq and Valkey are deleted. Local dev runs the same
   agent against localhost.
4. **Multi-step is in scope.** Orchestration comes from Temporal, not a
   bespoke DAG engine — but Temporal runs only on trusted infra (see layered
   architecture). Rejected alternatives: all-in Temporal (untrusted workers
   need Temporal Cloud or a custom authorizer, per-collaborator namespaces,
   payload codecs), bespoke `depends_on` DAG (grows into a homemade workflow
   engine).

## Architecture: brain and edge

**Edge (phase 1) — HTTPS runner protocol.** The queue is Postgres: a `Run`
row with `status=pending` and a non-NULL `lane` column is the queue entry
(there is no separate QUEUED status — `pending` already plays that role in
the existing lattice, and `lane IS NULL` marks a run not yet enqueued).
Runner agents poll `/api/v1/runner/*` endpoints to claim a run, fetch its inputs,
report progress, upload artifacts, and mark it terminal. `RunTraining` and
`RunPrediction` execute **unmodified** on the runner: they depend only on the
four ports (`RunRepository`, `DatasetRepository`, `ProtocolRepository`,
`BlobStore`), and the agent injects HTTP-backed implementations. Cancellation
needs no new mechanism — the training reporter already re-reads the run row
periodically; that read now travels over HTTPS.

**Brain (phase 2) — Temporal on trusted infra only.** Workflow code (sweeps,
ensembles, pipelines) runs on orchestration workers we own, colocated with
the backend. A workflow step needing compute enqueues a Run into the Postgres
queue via an activity and waits; the run-complete endpoint signals the
workflow to advance. Collaborator runners never connect to Temporal and don't
know it exists. Datasets/artifacts never pass through Temporal (it has ~2MB
payload limits); they flow through the blob endpoints. The existing
`JobEnqueuer` port is the seam between phases: phase 1 keeps "create one
pending run", phase 2 swaps in "start a workflow that creates N runs".

## Data model

New `runners` table (instance-level, not per-workspace):

| column | notes |
|---|---|
| `id` | uuid |
| `name` | display name, e.g. `marseille-gpu-01` |
| `lanes` | lanes this runner serves; source of truth is this row, not runner config |
| `token_hash` | SHA-256 of the bearer token; plaintext shown once at creation |
| `created_at`, `last_seen_at` | `last_seen_at` stamped on every authenticated call |
| `revoked_at` | nullable; set → every subsequent request 401s |

`runs` gains: `lane` (persisted at enqueue; today it is computed at enqueue
time and passed to arq), `claimed_by` (runner id), `lease_expires_at`,
`attempts`.

Migration backfills `lane='default'` on pending and running rows (arq's
in-flight queue disappears with Valkey; a wrong-lane pending run at cutover
re-runs on the default lane — slower, not wrong, matching how the gpu lane
already runs on CPU locally) and stamps running rows with an already-expired
lease so the first sweep requeues them. Terminal rows keep `lane` NULL (inert).

## Runner protocol

All endpoints under `/api/v1/runner/*` (matching the codebase's `/api/v1`
route convention), authenticated by per-runner bearer token
(constant-time hash compare, revocation checked per request). TLS terminates
at the reverse proxy. Machine tokens are separate from Sentinel user auth.

| endpoint | behaviour |
|---|---|
| `POST /api/v1/runner/claim` | Requeue expired leases (lazy sweep, no background task), then `SELECT … FOR UPDATE SKIP LOCKED` one pending run matching the runner's registered lanes, honouring the per-workspace concurrency cap. Sets `claimed_by` + lease, returns run payload, or 204. Plain poll (~3s client-side); no long-poll. |
| `GET /api/v1/runner/runs/{id}` | Run payload — backs `RunRepository.get_by_id`, which is also how the reporter detects cancellation. |
| `POST /api/v1/runner/runs/{id}` | Run update — backs `RunRepository.update`. PATCH-like: only the fields present in the request body are applied (a bare status heartbeat must not null out a previously-reported `phase`/`result_uri`/`error_message`); `protocol_id` routes through `Run.link_protocol` (write-once, 409 if already linked elsewhere). What's enforced: the caller must hold the current lease (fencing) — 403 otherwise; the run must still be active (status `pending` or `running`) for any write to be accepted at all; `status` must be one of `running`/`ready`/`failed` — `cancelled` is rejected outright, since cancellation is a user action, never a runner-reported one; `expected_version` must match the run's current version or the write is a 409. What's *not* enforced: the full domain transition lattice is not re-validated server-side, so a runner can report `ready` directly without ever having reported `running` first — accepted for phase 1 because the claim-holder could produce the same end state either way; flagged here as the known gap rather than silently relied upon. |
| `GET /api/v1/runner/runs/{id}/dataset` | Backs `DatasetRepository.get`, scoped to the claimed run's own dataset. |
| `GET/POST /api/v1/runner/runs/{id}/protocol` | GET backs `ProtocolRepository.get` (prediction's input protocol). POST backs `ProtocolRepository.add` — training *creates* the trained protocol row as its output; its `status` is forced to `DRAFT` server-side regardless of what the envelope carried, since publishing is a human, editor-role action elsewhere. Same run scoping. |
| `GET/PUT /api/v1/runner/runs/{id}/blobs/{key}` | Backs `BlobStore.get_bytes`/`put_bytes`. Every blob key in the codebase starts with `{workspace_id}/` (snapshot, upload, artifact, scorecard, predictions), and training writes artifact keys addressed by a protocol id it generates mid-job — so the enforceable v1 scope is the workspace prefix: GET and PUT both require `key.startswith(f"{run.workspace_id}/")`, PUT additionally requires the run to still be active, with a server-side size cap. Coarser than per-record ACLs; acceptable under "one lane = one trust domain", tightened by runner groups if lanes ever mix owners. |

The runner-side port surface was verified against the handlers:
`RunRepository.get_by_id`/`update`, `DatasetRepository.get`,
`ProtocolRepository.get`/`add`, `BlobStore.get_bytes`/`put_bytes`. The HTTP
port implementations cover exactly these methods; everything else on the
ports is API-side only.

Lease semantics: every run-scoped call extends the lease, so the reporter's
periodic re-read doubles as the heartbeat. Lease TTL 10 minutes — generous
because an engine that never reports is a known pre-existing gap (arq's hard
timeout was the old backstop). On expiry: `attempts < 3` → back to pending
(the fencing makes any late writes from the presumed-dead runner 409), else
FAILED. `Run.start()` already treats a redelivered RUNNING row as
restart-from-zero.

Fairness: the claim query enforces a per-workspace cap on concurrently active
runs (a predicate, not a system), so one user's sweep cannot starve other
workspaces. "Active" counts `running` rows *and* claimed-but-still-`pending`
ones — a claim leaves a run `pending` with `claimed_by` set until the runner
reports back `running`, so counting only `running` would let two concurrent
pollers both slip under the cap during that window. The cap is a setting
(`STUDIO_WORKSPACE_MAX_ACTIVE_RUNS`, default 10).

## Runner agent

Same backend package, new entrypoint (`python -m daikonstudio.infrastructure.runner`).
Config is exactly two env vars: `STUDIO_URL`, `STUDIO_RUNNER_TOKEN`. Lanes
come from the server-side runner record — `STUDIO_WORKER_LANE` is retired,
along with `STUDIO_WORKER_MAX_JOBS` and `redis_url`. One job at a time
(preserves the GPU-memory rule `STUDIO_WORKER_MAX_JOBS=1` used to encode).
Loop: claim → build handlers with HTTP ports → run → terminal update → claim
again. A claim-endpoint error (non-2xx, or the studio unreachable at all)
does not end the agent: it is logged and treated the same as an empty claim,
retried on the next poll — nothing in this repo supervises or restarts the
process, so the loop itself has to be the thing that survives a studio
mid-deploy or mid-restart.

Packaging: one Dockerfile, two images — `daikon-runner:cpu` and
`daikon-runner:gpu` (CUDA base). This finally produces the GPU image that has
been pending. `make dev` starts two local agents (default + gpu lane) against
localhost instead of two arq workers, after `make seed-runners` (also run
from `make up`) ensures the two local dev runners those agents authenticate
as (`dev-local-default`/`dev-local-gpu`, fixed dev tokens) exist in the
database; `STUDIO_INLINE_JOBS=1` remains for tests and worker-less dev.

## Security

- Per-runner revocable bearer tokens, SHA-256 at rest, shown once.
- Lease fencing on every run-scoped write.
- Blob GET/PUT confined to the claimed run's workspace prefix
  (`key.startswith(f"{run.workspace_id}/")`), rejecting any key with a `..`
  or `.` segment or a leading `/`; PUT additionally requires the run still
  be active and enforces the upload size cap while streaming, so an
  unbounded chunked body without `Content-Length` still gets cut off.
  Nothing a runner uploads is ever executed on the server.
- **Trust rule (documented, not code): one lane = one trust domain.** Trained
  artifacts are executable (pickles/checkpoints), so whoever runs prediction
  with a protocol must trust whoever trained it. Do not mix owners on a lane.
  Upgrade path: runner groups.
- Runner management requires the editor role via Sentinel: `CreateRunner`
  and `RevokeRunner` both call `require_editor`, so an authenticated viewer
  gets 403. Listing runners only requires an authenticated caller (any role).
- No rate limiting in v1: authenticated machine endpoints, single-digit
  runner count expected.

## UI

One **Runners** page: table (name, lanes, online/offline derived from
`last_seen_at` freshness, current run), **New runner** dialog (name + lanes →
token shown once inside a copy-paste `docker run` command), **Revoke**
button. Uses the existing generated-API-client flow (`make generate-api`).

## Deletions

`arq` dependency; `ArqEnqueuer` and `WorkerSettings` in
`infrastructure/worker.py`; Valkey service in docker-compose; `redis_url`
setting; `WORKER`/`WORKER_GPU` Makefile blocks. `test_lanes.py` was not
deleted wholesale (reviewed deviation from the original plan): its
arq-specific tests were removed and its ~13 use-case lane-routing tests
(which engine's manifest routes a run to which lane) were kept, since
queue-level lane mechanics moved to `test_run_queue.py` against the real
`SqlAlchemyRunQueue` but use-case-layer routing is a different concern arq's
removal never touched. `InlineEnqueuer` survives unchanged.

## Testing

- Unit: claim semantics (lane filter, SKIP LOCKED contention, lease expiry →
  requeue, attempts cap, per-workspace fairness cap), stale-lease writes
  rejected, token auth (hash compare, revocation), the update route's actual
  invariants (lease fencing, active-status requirement, `cancelled` rejected,
  version match) — not the full domain transition lattice, which phase 1
  deliberately does not re-derive server-side (see the runner protocol table).
- Integration: the existing full-loop test runs the agent in-process against
  the test app through the HTTP ports — proving the unmodified-handler claim
  end to end.
- Phase 2 adds: workflow-level tests with Temporal's test environment
  (time-skipping), and an activity↔queue integration test.

## Phasing

1. **Phase 1 — the edge.** Postgres queue, runner protocol, agent, Docker
   images, Runners UI, arq/Valkey removal. Collaborators can attach GPUs and
   train immediately. Single-step runs only.
2. **Phase 2 — the brain.** Temporal dev-server in docker-compose,
   orchestration worker, first multi-step feature (e.g. hyperparameter
   sweep), enqueuer seam swapped to start workflows. Production Temporal
   self-hosting decisions land here, backed by the existing Postgres.

This spec fixes phase 2's architecture (Temporal on trusted infra behind the
enqueuer seam) but not its details; phase 2 gets its own spec and
implementation plan when it starts. The implementation plan that follows
this document covers phase 1 only.

## Out of scope (deliberate, with upgrade paths)

- Runner groups / per-dataset ACLs — lane = trust domain until then.
- Long-polling or websockets — plain 3s poll is noise next to a fit.
- Public/community runners — needs sandboxing + result verification.
- Auto-scaling and agent auto-update.
