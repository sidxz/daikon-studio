# Handoff — daikon-studio, after fan-out sweeps landed

**Written:** 2026-08-05, at the end of the fan-out sweeps build.
**For:** whoever picks this branch up next — could be a session finally starting Temporal, or
one that just wants to build on top of sweeps.
**This is not a plan.** It is the state of the world, what got verified live and what did
not, and the traps that are still there. Read `HANDOFF-chemprop-and-baselines.md` for the
era before self-hosted runners. The self-hosted-runners era's own state (registration,
lanes, the reverse-proxy caveat) is folded into section 4 below rather than a separate file.

---

## 1. Where things are

Branch **`self-hosted-runners`**, still **unpushed and unmerged**, stacked on
`baselines-and-pretrained` (itself unmerged). HEAD is `0fba723` (`feat(sweeps-ui): submit
form`) plus this handoff's own commit on top. The entire feature exists only on this laptop.
**Do not merge or push without the user asking** — that instruction from the previous
handoff still holds; nothing about finishing this plan changes it.

Working tree is otherwise clean apart from the untracked
`docs/superpowers/2026-08-04-engine-roster-research.md`, which belongs to someone else —
leave it alone.

### What this plan added

A **`sweep_id` column** on `runs` (nullable, indexed with `workspace_id`) and a **grouped
UI** on top of it — not a workflow engine, not a new queue, not a change to the runner
protocol. Concretely:

- `SubmitSweep` (`application/execution/sweeps.py`) creates N training runs sharing one
  `sweep_id` by calling the existing `TrainProtocol` N times. Each run keeps its own
  mandatory baseline — twenty configs against a shared baseline would be a real dependency,
  which is exactly what this feature was scoped to avoid.
- `CancelSweep` cancels every non-terminal member and reports how many; idempotent by
  construction (`Run.cancel()` raises on an already-terminal row, `CancelSweep` catches that
  and skips rather than propagating).
- `GET /api/v1/sweeps`, `GET /api/v1/sweeps/{id}`, `POST /api/v1/sweeps/{id}/cancel` —
  three new routes, `interface/routes/sweeps.py`.
- Frontend: `frontend/src/features/sweeps/` — a submit form (N config rows, each an
  engine + its resolved conditions), a ranked detail table (best first, signed baseline
  deltas, direction-aware), and a list page. Ranking direction is read from the metric
  convention already shared with Scorecards (`primary_metric_for`), which is what makes the
  detail table's top row match the Scorecard's own number for that run — proven live, see
  below.
- One optional `metrics` field added to the existing runner update envelope. That is the
  entire runner-facing surface of this plan; `_CLAIM` and the rest of the queue never
  learned anything about sweeps.

### Gates at HEAD (all read live during Task 10, not remembered from an earlier run)

- `make test-all`: **522 passed**, import-linter **3 kept, 0 broken** (Clean Architecture
  layers, domain purity, bounded-context independence).
- `make lint`: ruff check clean, ruff format clean (206 files), mypy clean (134 source
  files).
- `make lint-fe`: biome clean (131 files checked, no fixes applied).
- `make test-fe`: **67 passed** across 9 test files (up from 51 at the previous handoff —
  the sweeps feature's own `rank.test.ts` and `sweep-form.test.tsx` account for the delta).

---

## 2. The decision this plan did NOT touch: Temporal is still deliberately held

Nothing in this plan reopens that question, and the trigger to revisit it is **unchanged**
from the previous handoff:

> The first time step N+1 must consume step N's output and survive a crash in between.

A sweep does not cross that line — every run is independent, nothing waits on another run's
output, and a crash mid-sweep just leaves some runs at whatever status they reached; nothing
needs to be replayed as a unit. **Ensembles and featurize→train→evaluate pipelines do cross
it** — those are the concrete shapes of work that would justify Temporal, not sweeps. If you
are reading this because you're about to build one of those, that is the actual trigger;
don't back into Temporal for something a `sweep_id` column would have covered.

The `JobEnqueuer` port is still the seam a workflow engine would plug into, same as before.

---

## 3. What was verified live, and what was not

**Verified live, this session, against the real stack** (backend `:8002`, frontend `:3003`,
Postgres `:5435` via docker compose, both runner agents restarted on current code):

- **A real 3-config sweep end to end** (Task 9's work, confirmed still standing): BBBP,
  `ecfp4-randomforest` at two `n_estimators`, `ecfp4-xgboost` at defaults. Ranked table came
  back MCC 0.632 / 0.603 / 0.596 descending, signed baseline deltas, each row's Scorecard
  link matching the ranked table's number for that row — the proof `primary_metric_for` is
  genuinely shared, not coincidentally similar. `SELECT sweep_id, metrics FROM runs WHERE
  sweep_id IS NOT NULL` showed populated `metrics` for every `ready` row, confirmed again
  this session against every sweep submitted (11 `ready` rows across 5 sweeps, all with real
  `{value, baseline_value, primary_metric}` JSON — the runner wire path, not
  `InlineEnqueuer`, since `make dev`'s agents were used throughout).
- **The cancel cascade, three separate times, against three separate sweeps.** In every
  case: runs still `pending`/`queued` at the moment of cancel went straight to `cancelled`
  with `metrics` left as JSON `null` (confirmed by direct SQL, not just the UI badge); any
  run that had already reached `ready` before the cancel request's own DB read kept its
  `ready` status and real metrics untouched. A second `POST .../cancel` against an
  already-fully-terminal sweep returned **204** every time (confirmed in the backend's own
  access log, not just the UI) — consistent with the route's unconditional `status_code=204`
  and `CancelSweep`'s per-run `try/except DomainError: continue`, which makes a sweep with
  zero cancellable runs left a no-op rather than an error.
- **A genuinely surprising timing result, worth carrying forward:** on this BBBP-sized
  dataset (~2,000 rows), `ecfp4-randomforest`/`ecfp4-xgboost` complete their *entire*
  handler — chosen fit, baseline fit, optimism-gap fit, predict, Protocol write — in
  roughly 1–4 seconds *even at each engine's maximum hyperparameters* (2000 trees / 2000
  boosting rounds, depth 20). `Run.cancel()` does flip a `running` row's status to
  `cancelled` immediately at the DB level (verified by reading the domain code, not
  guessed) — but the code path that would make that visible as a live "Running →
  Cancelled" UI transition requires my cancel request's own read of the run list to land
  *before* the row's natural completion write does. Across three attempts (including one
  submitted and cancelled in the same browser-automation batch with no intervening wait), I
  never won that race — the run I'd caught as "Running" in one screenshot had already
  become "Ready" by the next. This is not a bug: it is the correct, race-free behavior
  (`run_job` re-reads before calling `succeed()`, so a cancellation that lands even a moment
  after the last checkpoint is never silently overwritten with `ready` — see the docstring
  on that re-read in `infrastructure/jobs.py`). It does mean: **if you need to *demonstrate*
  a mid-flight cancellation interrupting a running fit, fast CPU engines on a small dataset
  cannot show it** — only an engine that calls `ctx.report()` per step (chemprop is the only
  one that does today) gives cancellation a checkpoint to land on before the whole handler
  finishes. `RunTraining._checkpoint`'s own docstring already explains this; it just hadn't
  been exercised against a live sweep before this session.

**Not verified, and worth flagging rather than silently skipping:**

- No live sweep was submitted with `chemprop-dmpnn` as a config. The brief steered away from
  it deliberately (single server-wide `STUDIO_WORKER_JOB_TIMEOUT`, no per-lane override —
  see section 4), so mixed-engine sweeps involving the GPU lane remain untested end to end,
  though nothing in `SubmitSweep` special-cases the engine and the per-run enqueue path is
  identical regardless of lane.
- No sweep near the `MAX_CONFIGS = 50` cap was submitted; the largest tested was 6.
- The `aria-pressed` gap on the Runners page's lane pills (section 4) was not touched by
  this plan and was not re-checked this session.

---

## 4. Open items carried forward, unchanged, from the previous handoff

None of these were in this plan's scope, and this plan did not touch them:

- **`Dockerfile.gpu` has still never been built.** arm64 laptop, cu124 wheels are amd64-only.
  The CPU image builds fine (`make image-runner-cpu`).
- **Per-lane job timeouts still do not exist.** `worker_job_timeout` is served from the
  server to every runner regardless of lane; raising it for chemprop raises it for every
  fast CPU config too. This is exactly why sweeps steer clear of chemprop configs by
  convention rather than by any enforced limit — nothing stops someone from putting a
  chemprop config in a sweep, it is just not the sweep this plan's own live testing used.
- **The reverse-proxy `merge_slashes` caveat, unresolved, before the first non-local
  runner.** Blob keys can arrive as full `file:///…` store URIs; nginx's default
  `merge_slashes on` collapses `file:///` → `file:/`, the prefix strip misses, and every
  prediction artifact read 403s. Direct-to-uvicorn (which is all this session used) is
  unaffected. Fix before deploying a runner anywhere but this laptop.
- **The runner trust boundary is semi-trusted, not untrusted.** Any registered runner can
  still claim from every workspace, a blob capability still outlives its lease, trained
  artifacts are still executable, and nothing verifies a runner told the truth. Fine for
  people you'd give an account to; not for strangers. Sweeps add no new surface here — every
  run a sweep creates goes through the exact same claim/lease machinery as a run submitted
  by hand.
- **The Runners page's lane pills still lack `aria-pressed`.** Selection is invisible to a
  screen reader and genuinely hard to read visually. Worth fixing next time that page is
  touched; this plan never opened it.

---

## 5. New deferred items from this plan

Pulled straight from `.superpowers/sdd/2026-08-05-fanout-sweeps/progress.md`, stated
plainly rather than left buried in a ledger:

- **No test catches a regression that re-adds `sweep_id` to `SqlAlchemyRunRepository
  .update()`'s `.values()`.** The existing test sets `sweep_id` at construction and never
  mutates it afterward. A follow-up test that constructs a run, mutates `run.sweep_id`
  directly, calls `update()`, and asserts the column is unchanged in the database would
  close this. `sweep_id` is deliberately write-once by the same convention `params` already
  follows — see `Run.protocol_id`'s own comment for the argument.
- **`test_update_run_applies_metrics` only asserts HTTP 200 on its own.** It proves the
  runner update envelope *accepts* a `metrics` field, not that the server *applied* it. The
  paired read-back test in the same file does cover application, so the pair is sound
  together — but the single test would pass even if a future change silently dropped
  `metrics` on the way to the database, as long as the envelope still parsed. Worth
  strengthening if that test is ever touched again.
- **`SubmitSweep` costs an N+1 dataset read per sweep.** It pre-flights the dataset once to
  validate the request, then `TrainProtocol` re-fetches the same dataset once per config.
  This is pre-existing `TrainProtocol` behavior, left alone deliberately — the plan's own
  brief scoped it as "otherwise untouched." Worth revisiting if a sweep's config count grows
  large enough for this to show up in real latency; it did not for the 6-config sweeps this
  session ran.
- **`SweepDetailResponse.from_runs` carries an empty-list guard that is dead code on both of
  today's call sites.** Submit fails before creating anything on an empty request, and
  `GetSweep` returns `NotFoundError` before an empty list would ever reach this function.
  The reviewer judged this reasonable defensive code rather than premature abstraction and
  left it in; flagging so a future reader doesn't mistake "dead on today's call sites" for
  "wrong."
- **`rankRuns` reads metric direction from the left operand only.** This is correct *because*
  every run in one sweep shares a single task type by construction (`SubmitSweep` validates
  every config's engine against the same dataset/task), but that homogeneity assumption is
  not documented at the function's own boundary. A hypothetical mixed-direction list would
  sort non-transitively and silently. A one-line comment on `rankRuns` would close this;
  it was left as a known gap rather than blocking the plan on a comment.
- **The rank cell and `rankRuns` check a run's metric value two different ways.** The rank
  cell tests `metrics?.value == null`; `rankRuns` itself buckets on
  `typeof value === "number"`. These diverge only for a non-numeric, non-null value on that
  field, which the API never actually sends — so this is dormant, not live, but it is two
  different assumptions about the same `unknown`-typed field living in two places. This is
  the plan's own given code, not something the implementer introduced.

---

## 6. What not to do

- **Do not merge this branch anywhere without asking.** Still unpushed, still the user's
  call, unchanged from every previous handoff on this branch.
- **Do not start Temporal because sweeps exist now.** Section 2 restates the actual trigger;
  sweeps are evidence the trigger has *not* fired yet, not evidence it's close.
- **Do not teach the queue or the runner agents about sweeps.** They still don't know sweeps
  exist, and that was the entire point — `_CLAIM` in
  `infrastructure/persistence/sqlalchemy/execution/queue.py` is inside an
  adversarially-reviewed security boundary that this plan was explicitly told to leave
  alone, and did.
- **The runner protocol is a partial exception, and that's fine — don't "fix" it.**
  `RunEnvelope` (`infrastructure/runner/wire.py`) mirrors every `Run.__init__` kwarg by
  contract, so it picked up `sweep_id` for free the moment `Run` gained the column: the claim
  response and `GET /runner/runs/{id}` now carry it. A runner can only *read* that field,
  never write it — `RunUpdateEnvelope`, the only body a runner ever POSTs back, has no
  `sweep_id` field, so the column stays exactly as write-once from the runner's side as
  `params` already is. Do not add one "so the mirror is complete" — that would let a
  compromised or buggy runner re-point a run at a different sweep after the fact, which is
  the write-once property this is supposed to protect.
- **Do not "fix" `STUDIO_WORKSPACE_MAX_ACTIVE_RUNS`** because a 50-config sweep feels slow
  against it. That cap is the fairness predicate stopping one sweep from starving every
  other workspace on the instance; Task 8's UI makes the cap visible rather than raising it,
  which is the correct fix.
- **Do not read a "Running → Ready" race as evidence the cancel cascade is broken.** Section
  3's timing note explains why fast CPU engines on a small dataset will very often show this
  exact race, and why it is the *correct* behavior, not a bug to chase.
