# Handoff — daikon-studio, after self-hosted runners landed

**Written:** 2026-08-05, at the end of the self-hosted-runners build.
**For:** a fresh session building **fan-out (grouped runs / sweeps)**.
**This is not a plan.** It is the state of the world, the decision that was taken and
why, and the traps that cost real time. Write the design yourself — brainstorm first,
then `writing-plans`. Read `HANDOFF-chemprop-and-baselines.md` for the era before this.

---

## 1. Where things are

Branch **`self-hosted-runners`**, **26 commits, none pushed**, stacked on
`baselines-and-pretrained` (which is itself unmerged and unpushed — check before you
assume `main` is the base). The entire feature exists only on this laptop.

Working tree clean apart from the untracked `docs/superpowers/2026-08-04-engine-roster-research.md`,
which belongs to someone else — leave it alone.

Gates at HEAD: **`make test-all` 497 passed**, import-linter 3/3 contracts kept,
`make lint` clean (ruff + mypy, 132 files), `make lint-fe` clean, `make test-fe` 51 passed.

### What that branch did

arq and Valkey are **gone**. The `runs` table is now the job queue, and compute happens
on **runner agents** that hold nothing but a URL and a token:

| | |
|---|---|
| Backend `:8002`, Frontend `:3003` | native, via `make dev` |
| Runner agent — default lane | `python -m daikonstudio.infrastructure.runner`, token `drt_dev_default` |
| Runner agent — gpu lane | same module, token `drt_dev_gpu`, chemprop **on CPU** |
| Postgres `:5435` | docker compose (**no Valkey any more**) |
| Blobs | `.blobs/` at the repo root |
| Sentinel | remote, service name `daikon-studio-dev` |

`make up` migrates and seeds the two dev runners; `make dev` starts everything. Lanes now
live on the **server-side `runners` row**, not in any worker's environment —
`STUDIO_WORKER_LANE` is retired and naming it in a message or comment is now a lie.

Verified live end to end on 2026-08-05 through the browser: registration → one-time token
→ training on BBBP via the default-lane agent (`claimed_by = dev-local-default`,
`attempts = 1`) → scorecard (MCC 0.603 vs baseline 0.632) → publish → prediction of 5
compounds → revoke, after which the revoked token 401s immediately.

Read `docs/superpowers/specs/2026-08-04-self-hosted-runners-design.md` before touching
any of it. Its Security section is current and includes the reverse-proxy caveat below.

---

## 2. The decision you are inheriting

The spec's phase 2 was **Temporal**, orchestrating multi-step work on trusted infra with
runners staying the untrusted edge. **That is on hold, deliberately.** Do not start it.

The reasoning, so you can overturn it if you find it wrong rather than merely inherit it:

- "Many users, many models, many nodes" is **throughput**, and phase 1 already does that.
  N agents pull one queue, `SKIP LOCKED` stops collisions, the per-workspace cap stops one
  user hogging the fleet. Adding a node is a `docker run`. Temporal adds nothing there.
- What phase 1 genuinely cannot express is **dependency** — fan out 20 fits, wait for all,
  pick the winner, register it. Durable recovery of *that* state is what Temporal actually
  sells, and it is the part that is miserable to hand-roll.
- But a large slice of the value is **pure fan-out with no dependency**: submit N configs,
  watch them together, see the best. That needs a parent id and a grouped UI, not a
  workflow engine.

**So: build the fan-out. Leave Temporal until a real dependent workflow forces it.**
The trigger to revisit is concrete and worth writing on the wall: *the first time step N+1
must consume step N's output and survive a crash in between.* Ensembles and
featurize→train→evaluate pipelines cross that line. A sweep does not.

Nothing you build now is wasted either way — the `JobEnqueuer` port is still the seam a
workflow engine would plug into.

---

## 3. The ask, as stated

> "lets hand off to a fresh session to build the fan out. hold temporal for now"

That is the whole brief. It has **not** been through brainstorming, so the shape below is
context, not requirements — the user has not agreed to any of it.

Questions I would put to them first, because each changes the design:

1. **What varies across a sweep?** Engine conditions only (learning rate, depth…), or
   engine choice too, or the split seed (which is the replication story in
   `publication-grade-gaps`, a different feature wearing similar clothes)?
2. **Is the winner picked by the system or the human?** "Rank them and let me choose" is
   a grouped list. "Register the best automatically" needs a criterion, which needs a
   metric contract, which is most of a dependency.
3. **Does a sweep own its runs?** Cancelling a sweep — does it cancel 20 pending runs?
   That is a real cascade with real edge cases against the claim/lease machinery.
4. **How does the mandatory baseline interact?** Every training run today fits the chosen
   engine *and* a baseline. 20 configs is 40+ fits. Does the sweep share one baseline run,
   or pay for it 20 times? Sharing is a dependency; paying is honest but expensive.

---

## 4. Where the code will resist you

**The enqueue seam is one line, and it is the right one.**
`application/execution/train_protocol.py:358` —
`await self._enqueuer.enqueue(run.id, lane=lane_for(...))`, and the prediction twin at
`predict_with_protocol.py:236`. A sweep creates N runs and enqueues each. Nothing about
the queue, the protocol, or the agents needs to change for fan-out. Resist the urge to
teach the queue about groups.

**`Run.params` is write-once by convention.** `SqlAlchemyRunRepository.update` deliberately
never persists it, so a handler cannot rewrite its own instructions mid-flight. If a sweep
id lives in `params` it can never be corrected. A real nullable column
(`parent_run_id`, indexed with `workspace_id`) is the honest choice — that is exactly the
argument `Run.protocol_id`'s comment makes for itself
(`domain/execution/run.py:99-104`). Read it before deciding.

**`ListRunsQuery` (`application/execution/list_runs.py:40`) has `kind`, `protocol_id`,
`cursor`, `limit`** and keyset-paginates on `(created_at, id)`. Grouping by parent is a
filter here, not a new listing path.

**The per-workspace concurrency cap will throttle your sweep, by design.**
`STUDIO_WORKSPACE_MAX_ACTIVE_RUNS` defaults to 10 (`queue.py` `_CLAIM`). A 20-run sweep
does not run 20-wide; it drains 10 at a time. That is the fairness predicate doing its
job — do not "fix" it because a sweep felt slow, or one user's sweep starves the instance.

**The runner agents do not hot-reload.** Change anything under
`infrastructure/engines/*` or `application/execution/*` and you must
`make dev-worker` / `make dev-worker-gpu`, or you will debug a fix that never loaded.
This has already cost a full debugging detour once.

**Tests that touch `runners` rows must clean up.** Fixtures bound to the session-scoped
NullPool engine have no per-test rollback; a leaked row deterministically breaks
`test_runner_repository.py::test_list_returns_all` when the whole suite runs in one
session (which `make test-all` does). Use `cleanup_registered_runners` in
`backend/tests/helpers/runner_fixtures.py`.

**Frontend:** copy `frontend/src/features/runs/` structure; nav is one entry in
`shared/lib/navigation.ts`; run `make generate-api` and commit the regenerated client.

---

## 5. Open items you are inheriting

Not blockers for fan-out, but they are real and undocumented elsewhere:

- **Reverse-proxy caveat (do this before the first non-local runner).** Blob keys can
  arrive as full `file:///…` store URIs, so nginx's default `merge_slashes on` collapses
  `file:///` → `file:/`, the prefix strip misses, and **every prediction artifact read
  403s**. Direct-to-uvicorn is unaffected. Fix: disable slash merging for
  `/api/v1/runner/`, or normalize client-side in `HttpBlobStore` too
  (`infrastructure/runner/ports.py`). Written up in the spec's Security section.
- **`Dockerfile.gpu` has still never been built** — arm64 laptop, cu124 wheels are
  amd64-only. The CPU image builds fine (`make image-runner-cpu`).
- **Per-lane job timeouts no longer exist.** `worker_job_timeout` is served from the
  server to every runner regardless of lane. Raising it for chemprop raises it for
  everything.
- **Trust boundary is semi-trusted, not untrusted.** Any registered runner can claim from
  *every* workspace (`_CLAIM` has no workspace-membership predicate), a blob capability
  outlives its lease (`_VERIFY` checks `claimed_by` but not `lease_expires_at`), trained
  artifacts are executable, and nothing verifies a runner told the truth. Fine for people
  you would give an account to; not for strangers.
- **A11y, worth fixing when you next touch the Runners UI:** the lane pills carry no
  `aria-pressed`, so selection is invisible to a screen reader and genuinely hard to read
  visually (light-grey fill vs white-with-border). Hit firsthand during live testing.
- Smaller deferred items are listed in `docs/superpowers/phase-1-deferred-items.md`
  and in the git history of this branch's review commits.

---

## 6. What not to do

- Do not start Temporal. Section 2 says why, and names the condition that would change it.
- Do not teach the queue, the runner protocol, or the agents about sweeps. Fan-out is a
  parent id and a UI; the machinery underneath is already correct and adversarially
  reviewed. Every line you add there is a line inside the security boundary.
- Do not put the sweep id in `Run.params`.
- Do not merge this branch anywhere without asking — the user chose "keep as-is" on
  2026-08-05 and it is still unpushed.
