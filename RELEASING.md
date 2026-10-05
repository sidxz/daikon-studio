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
   make security-scan   # Trivy: lockfiles, secrets, then both images
   ```
   It fails on any fixable HIGH/CRITICAL finding. Fix the dependency or base
   image; only add an id to `.trivyignore`, with the reason, when the code
   cannot run or the fix is a scheduled major. Never tag over a red scan.
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
