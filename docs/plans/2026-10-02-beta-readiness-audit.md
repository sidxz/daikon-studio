# Beta-readiness audit, 2026-10-02

The findings `2026-10-02-beta-readiness.md` closes. Audited against the Phase 1 product
spec, the frontend spec, the self-hosted-runners spec and `docs/roadmap.md`, by reading the
code and by running the gates, the test suites, the live dev stack and the Docker images.

## Verdict

Not ready to deploy a beta. The application is complete for Phase 1 and well beyond it;
what is missing is the deployment layer and a short list of day-one failures. For a
one-lab, one-workspace beta the blockers are each hours to a day of work. For a shared
instance across labs the runner trust model is a hard stop until it changes.

## Against the requirements

- Phase 1 success criteria 1 and 4 met as far as code and tests show; criterion 2 (a second
  user runs a published Protocol) implemented and workspace-scoped but exercised with one
  account only; criterion 3 half met: the negative verdict is clear, the positive verdict is a
  naked delta with no confidence interval, unsupported at n≈200.
- Shipped beyond Phase 1: seven engines on two lanes, self-hosted runners with lease fencing,
  fan-out sweeps, choosable baseline, CheMeleon pretrained weights, dataset diagnostics,
  server-side result sort and filter, a Runners page.
- Approved and not built: retry a failed run, per-lane job timeouts, an identifier column
  through predictions, prospective validation, lineage canvas, Butina/UMAP/temporal splits,
  multi-task training, transfer from a prior artifact.

## Blockers

1. **The only deployable image crashes at boot.** `backend/Dockerfile` builds green; the
   LightGBM engine imports at module load and needs `libgomp.so.1`, absent from
   `python:3.13-slim`. Verified by building and running; `libgomp1` fixes it. The same file is
   the default-lane runner image. The GPU image was built 2026-08-06; the lock changed
   2026-08-07 (transformers) and 2026-08-15 (duar-auth), so it cannot run `molformer-xl`.
2. **"Deploy" is not defined.** No frontend Dockerfile in the tree or history, compose has only
   Postgres, no CI, no image runs migrations, no reverse-proxy config. The RDKit wasm reaches
   `public/` only via a `|| true` postinstall.
3. **Sessions break after one hour.** Google ID tokens expire in an hour; the backend then
   401s every call; the Duar SDK's `getAuthState` never checks that token's expiry;
   `customInstance` has no 401 handling. The user sees errors or skeletons until a reload.
4. **Production would be blind.** No logging configured anywhere in the API (only WARNING and
   above reach stderr via `lastResort`); `/health` is unconditional; no request ids;
   `InlineEnqueuer` suppresses every exception silently.
5. **Runner trust is instance-wide.** Any editor in any workspace mints a runner token; the
   claim query filters by lane only; the blob guard is workspace-prefix, not run-scoped;
   artifacts are loaded with `pickle.loads`. Accepted by the runners spec as "one lane = one
   trust domain": fine for one lab in one workspace, a cross-workspace code-execution path
   on a shared instance.

## Fix before the first real scientist

- Compound identifiers do not survive prediction: results carry canonical SMILES, readouts,
  uncertainty and applicability only; unparseable rows are dropped silently, so even row
  order cannot join back to the upload.
- Dataset validation accepts inputs that fail later: null targets pass and training fails
  with `Input y contains NaN`; a numeric target holding `NA` is read as text and raises an
  unhandled 500; binary labels are never checked to be 0 or 1.
- UI dead ends: a finished training run opened from the Runners page renders a results grid
  whose request always 404s; run and sweep detail show a skeleton forever on error and keep
  polling; a cancelled training run hangs the form; login errors redirect with a message the
  login page never shows; every list except Datasets stops at the first page of 50.
- Deadlines are mostly theoretical: only chemprop and MoLFormer check the job deadline; tree
  and GP engines, predictions and inline mode have none; nothing hard-kills a hung fit; a
  gpu-lane run with no gpu runner sits pending forever and neither the engine nor the run
  response exposes the lane.
- Failed runs show `repr(exc)` to every workspace member, leaking blob paths and URLs.
- `alembic/env.py` omits the runners model, so the next autogenerate would drop the table.

## Hygiene

Seven uncommitted files (port move, Duar scope warning) and a test still named
`test_sentinel_scope.py` that fails the formatter; the dev venv lost the gpu and s3 extras so
`make lint` fails on mypy and both local runners advertise engines they cannot import; two
biome errors in the auth config file; OpenAPI snapshot drift on the runner wire schema; a
README that still says no frontend exists.

## Verification evidence

| Check | Result |
|---|---|
| ruff, import-linter | pass |
| mypy | fails, gpu extra missing from venv |
| biome | 2 errors |
| tsc, next build | pass, 26 routes |
| pytest unit | 260 passed, 2 skipped |
| pytest api + integration | 297 passed |
| vitest | 69 passed |
| uv and pnpm locks | in sync |
| alembic head vs live DB | both at 010 |
| OpenAPI snapshot | 1 schema missing, 1 differs |
| backend image | builds, crashes at boot |
| live dev stack | healthy, protected routes 401, realm scope resolved |
