# Releasing DAIKON Studio

The frontend and backend are versioned **independently** with **SemVer**, and
the **git tag is the single source of truth**. CI builds the image, injects the
version, and publishes a changelog + GitHub Release.

## Tag namespaces

| Component | Tag form          | Image                                          |
|-----------|-------------------|------------------------------------------------|
| Backend   | `backend-vX.Y.Z`  | `ghcr.io/sidxz/daikon-studio/api:X.Y.Z`        |
| Frontend  | `frontend-vX.Y.Z` | `ghcr.io/sidxz/daikon-studio/frontend:X.Y.Z`   |
| GPU runner | `backend-vX.Y.Z` | `ghcr.io/sidxz/daikon-studio/runner-gpu:X.Y.Z` (step 6, not CI) |

The backend image is both the API and the default-lane (CPU) runner, so a
backend release versions both. Pushing a tag in one namespace builds, tags, and
releases **only** that component, and moves its `:latest`. Pushes to `main` and pull
requests run the tests only and build no images.

## Choosing the bump (Conventional Commits)

Look at the commits since the component's previous tag:

| Commit type                          | Bump   |
|--------------------------------------|--------|
| `fix:`                               | patch  |
| `feat:`                              | minor  |
| `feat!:` / `BREAKING CHANGE:` footer | major  |

A backend **major** bump signals a breaking API change — the moment to check the
frontend and any deployed runners are compatible.

## Cutting a release

The test suites run once per commit, on `main`; the tag reuses that result. From a
green `main` to published images takes about 10 minutes, so release a batch of
changes at once rather than each fix as it lands.

1. Push to `main`. Its CI runs both suites (about 5 minutes). Before pushing, the
   static checks and the tests for what changed are enough locally; the full suite
   is CI's job.
2. While that CI runs, run the security gate locally:
   ```bash
   make security-scan   # Trivy: lockfiles, secrets, then every published image
   ```
   It fails on any fixable HIGH/CRITICAL finding. Fix the dependency or base
   image; only add an id to `.trivyignore`, with the reason, when the code
   cannot run or the fix is a scheduled major. Never tag over a red scan.

   Two things about this gate are worth knowing before you trust a number from it.

   It covers the chem runner image as of 2026-10-25, and did not before. That image
   is published and had never been scanned, which is how eight findings sat in it
   unseen; the findings were not the hole, the absence of a scan was.

   `trivy fs` over a directory of jars reports zero findings whether they are current
   or years old, because it does not detect them. Scan the built image. A zero from a
   directory scan means the scanner did not look, not that there is nothing there.

### The p2rank findings, open as of 2026-10-25

The chem image is **red on eight findings and must not be published** until someone
decides what to do about them. All eight are in jars that ship inside p2rank 2.5.1:
`jackson-core` 2.13.4 (CVE-2025-52999, CVE-2026-68494, CVE-2026-89425),
`jackson-databind` 2.13.4.2 (CVE-2026-54512, CVE-2026-54513, CVE-2026-91776,
CVE-2026-91777) and `msgpack-core` 0.8.24 (CVE-2026-21452).

We do not build those jars. p2rank is fetched as a released tarball pinned by
checksum, and `bin/lib` is its own. Upgrading does not help: 2.6-alpha scanned as an
image carries eighteen findings rather than eight, bundling more jars and a jackson
still behind its own advisories. It was tried and reverted. The alpha is otherwise a
faithful substitute, for what that is worth later: on 1HSG and 1CA2 it returns the
same rank 1, the same score and probability, the same centre and the same lining
residues as 2.5.1.

**Decided 2026-10-25: option 2.** The eight ids are suppressed in
`.trivyignore.yaml` with the reasoning above and an expiry of 2026-12-31, so the gate
passes and the expiry forces the question again rather than letting it rot. The three
options, kept for whoever revisits:

1. Wait for a stable p2rank after 2.5.1 that bundles current jackson.
2. Accept the risk and record it in `.trivyignore` against these eight ids. The case
   for accepting: p2rank runs as a subprocess over a prepared PDB this product wrote,
   its jackson parses p2rank's own configuration and model files, and nothing routes
   scientist-supplied input into it. The denial-of-service entries cost a failed step;
   the deserialisation entries need attacker-controlled polymorphic JSON that never
   arrives. That argument dies the moment p2rank is handed untrusted JSON or put on a
   network, so any such entry should carry an expiry and a date, not sit forever.
3. Drop p2rank and keep fpocket as the only pocket finder.

Whoever picks one should write the choice here.
3. Pick the next version per the table above. Inspect what changed:
   ```bash
   # commits touching the backend since its last tag
   git log "$(git tag --list 'backend-v*' --sort=-creatordate | head -1)"..HEAD -- backend/
   ```
4. Once `main`'s CI and the scan are both green, tag every component that changed and
   push the tags together:
   ```bash
   git tag backend-v1.4.0 && git tag frontend-v2.1.0
   git push origin backend-v1.4.0 frontend-v2.1.0
   ```
   Wait for `main`'s CI to pass first: a tag whose commit already passed there skips
   the suites. Tagged sooner, it runs them itself, about 6 minutes longer.
5. CI (`.github/workflows/ci.yml`) then:
   - finds `main`'s passing run for the tagged commit and skips the suites (the
     `tested` job), or runs them when there is none;
   - builds **only** that component, boot-checks the
     image, and tags it `1.4.0`, `1.4`, `1` and `latest` (a pre-release such as
     `1.4.0-beta.1` gets only its exact tag);
   - injects `APP_VERSION` / `APP_GIT_SHA` / `APP_BUILD_DATE` into the image — all three
     come from `scripts/build-info.sh <component>`, the single derivation shared with `make dev`;
   - generates a `git-cliff` changelog scoped to that component since its
     previous tag and publishes a **GitHub Release** ("Backend v1.4.0").

6. **Backend releases only:** publish the CUDA runner from the tag, right after pushing
   it, alongside the tag's CI. A worktree leaves your own checkout on `main`:
   ```bash
   git worktree add --detach ../release backend-v1.4.0
   make -C ../release publish-runner-gpu
   git worktree remove ../release
   ```
   It builds `backend/Dockerfile.gpu` on atlantic (x86_64, NVIDIA GPU, lasting layer
   cache) over the `atlantic` docker context, fails unless torch can see the GPU, runs
   the same Trivy gate on the image, and pushes `1.4.0`, `1.4`, `1` and `latest`. Not
   in CI: the image is ~22 GB, and atlantic is not a self-hosted GitHub runner because
   this repo is public, so any fork PR could target one. One-time setup: `docker login
   ghcr.io` on atlantic with a token that has `write:packages`.

To deploy a release, pin `STUDIO_API_IMAGE` / `STUDIO_FRONTEND_IMAGE` in
`deploy/.env` to the version instead of `:latest`.

### Deploy the API before the runners

**Always, including the GPU runner on atlantic.** Runners write the training blobs
the API reads back, and `ScorecardInputs.from_json` is `cls(**raw)` — so a blob from
a newer runner carrying a field an older API has never heard of raises `TypeError`
and the Scorecard 500s. The API tolerates an older runner's blob (every new field
has a `None` default); an older API cannot tolerate a newer runner's. The order is
therefore API, then default-lane runners, then atlantic's GPU runner.

This is not specific to any one release. It holds for every release that adds a
field to a stored blob, which most training changes do.

## Where the version shows up

- **App sidebar:** `UI v<version>`, from the baked image.
- **About card** (`/settings` → About): UI version + commit + build date, the
  live **API** version + commit + build date (fetched from `GET /version`), and
  the environment.
- **Backend `GET /version`:** unauthenticated JSON
  `{name, version, git_sha, build_date, environment}` — handy for `curl`/monitoring.

The GPU runner carries the backend's version: same code, a CUDA build of torch.

## Between releases

Builds between tags identify as the `git describe` form, e.g.
`1.4.0-128-g84e7848` — base tag, commits ahead, short sha. That is expected and
honest: it is not a clean release.

## Not the source of truth

`backend/pyproject.toml` and `frontend/package.json` carry a placeholder version
that **nothing displays**. The git tag is authoritative; there is nothing to bump
in those files when releasing. Locally, `make dev` exports the same identity the
image would get (`scripts/build-info.sh` → `git describe` against the nearest
`<component>-v*` tag, e.g. `0.1.0-3-g2b000f0`, plus the short SHA and a build
date); with no tag at all both apps show `0.0.0+dev` / `unknown`.
