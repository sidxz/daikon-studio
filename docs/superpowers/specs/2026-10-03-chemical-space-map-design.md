# Chemical space map

Date: 2026-10-03 · Branch: `chemical-space-map` (stacked on `explainer-figures`) · Status: design approved in conversation

## Goal

On a prediction run, show where the run's compounds sit relative to the protocol's training set. On a protocol, show where the training, validation and test compounds sit. Both views use real data. The map answers "where": which part of the training chemistry a compound is nearest to, and whether a run as a whole lands in new territory. The real Tanimoto similarity, drawn on top of the map, answers "how far". No 2D projection preserves fingerprint distances, so the map position must never be the only signal of novelty.

Training sets reach 50,000 to 100,000+ compounds (Chemprop, MoLFormer), so the renderer is WebGL.

## Decisions

| Decision | Choice | Source |
|---|---|---|
| Layout method | UMAP on ECFP4 (radius 2, 2048 bits), Jaccard (Tanimoto) metric | User |
| Run compound placement | Similarity-weighted center of its 5 most similar training compounds | User |
| Existing protocols | Backfill all now, with a one-off command | User |
| Placements | Run page and protocol page | User |
| Rendering | WebGL2, ported from ChemCellar's cluster-map renderer; no rendering library | User ("argue for WebGL") |
| Where the map is computed | Inside the training job, best-effort, after the Protocol row exists | Design: a new RunKind would touch ~11 places and add a deploy-order risk |
| Where neighbors are computed | Inside the prediction job, from the nearest-neighbor search it already runs | Design |
| Run results stay immutable | Neighbors go in a separate blob beside `predictions.parquet` | Design |

## Data

### Map blobs (per protocol)

- `{ws}/protocols/{pid}/chemical-space.parquet`: one row per dataset compound, in snapshot order. Columns `x` and `y` (float32, normalized into [0, 1] with the aspect ratio preserved), `partition` (int8: 0 train, 1 validation, 2 test), `structure` (canonical SMILES, for hover lookups).
- `{ws}/protocols/{pid}/chemical-space.json`: `{version, method: "umap", params: {n_neighbors, min_dist, metric, seed, fit_size}, umap_version, counts: {train, validation, test}}`.
- **Invariant:** the train rows of the parquet, in order, are exactly `ScorecardInputs.train_structures`. Both come from `frame.filter(split == "train")` in snapshot order. Neighbor indices refer to this train order.
- `MAP_VERSION = 1`. The backfill rebuilds a map whose stored version is lower.

### Neighbor blob (per prediction run)

- `{ws}/runs/{rid}/neighbors.parquet`: one row per row of `predictions.parquet`, in the same order. Columns `neighbor_index` (list[int32], up to 5, indices into the protocol's train order) and `neighbor_similarity` (list[float32], descending).
- Written only when the protocol's training structures are readable, which is the same condition under which `applicability` is non-null today.
- `applicability` becomes `neighbor_similarity[0]`, from the same single search, so the two can never disagree.

### Layout (`umap_layout`)

- ECFP4 bits as a sparse boolean CSR matrix, then `umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, metric="jaccard", random_state=seed)`. The seed is the dataset's `SplitSpec.seed`, the same one seed everything else uses. A fixed `random_state` makes UMAP single-threaded and reproducible; that is the accepted cost.
- Fewer than 5 compounds: no map. Status "too few compounds".
- **Size ceiling:** above 150,000 compounds, fit on a seeded 150,000 sample and place the rest by the same top-5 neighbor rule used for runs. Marked with a `ponytail:` comment.

### Nearest neighbors (`nearest_neighbours_tanimoto`)

- `nearest_neighbours_tanimoto(query, reference, k) -> (indices[n, k], similarities[n, k])`. It walks the query in bands of 128 rows, so memory is about `128 × len(reference) × 4 B` (51 MB at 100,000), not `len(query) × len(reference) × 4 B` (4 GB at 10,000 × 100,000, which is what the current code allocates).
- `nearest_neighbour_tanimoto` becomes the `k = 1` case of the same function. This is one implementation, so applicability, scorecard coverage and map placement share it.
- The `StructureNormalizer` port gains `nearest_neighbours_tanimoto`.

## API

All routes are workspace-scoped and authenticated, like their siblings.

- `GET /api/v1/protocols/{id}/chemical-space` → `{status: "ready", method, params, points: {x: number[], y: number[], partition: number[]}, counts}`, or `{status: "missing"}`. It is never a 404 for a missing map; it is a 404 only for a missing protocol. Coordinates are rounded to 4 decimals.
- `GET /api/v1/protocols/{id}/chemical-space/compounds?indices=…` (at most 50) → `[{index, structure, partition}]`, for hover.
- `GET /api/v1/runs/{id}/chemical-space` → `{status: "ready" | "missing", points: {x, y, row_id, applicability, neighbors: number[][]}, summary: {total, in_domain, threshold}}`.
  - Placement is computed server-side from the map and the neighbor blob.
  - `neighbors` are converted to map point indices so the client can draw lines to them.
  - `"missing"` means either the protocol has no map or the run has no neighbor blob.
  - 404 for a non-prediction run and 409 for a run that is not ready, like the results endpoint.
- `GET /api/v1/runs/{id}/chemical-space/compounds?rows=…` (at most 50) → `[{row_id, structure, compound_id, values, applicability}]`, for hover.

## Backfill

`python -m daikonstudio.infrastructure.maps.backfill`, plus a `make backfill-maps` target.

- Lists every protocol across workspaces with raw SQL, because repositories are workspace-scoped.
- For each protocol whose map is missing or out of date, it builds the map from the dataset snapshot.
- For each READY prediction run of that protocol without a neighbor blob, it computes neighbors from `predictions.parquet` and the scorecard inputs' train structures.
- It is idempotent, logs one line per item, and continues past failures. It exits non-zero if any item failed.
- In production it runs once, like `migrate`: `docker compose run --rm api python -m …`.

## Training and prediction changes

- **`RunTraining`:** after the Protocol row is added, report the phase "Mapping chemical space", build the map off the event loop, and write both blobs. Any exception is logged and swallowed; the protocol stands without a map.
- **Prediction:** one call to `nearest_neighbours_tanimoto(structures, train_structures, 5)` replaces the old top-1 call. Write `neighbors.parquet` after `predictions.parquet`.
- **`HttpBlobStore.get_bytes`** raises `FileNotFoundError` on 404, so the runner path degrades like the inline path. This is the root-cause fix in the shared adapter.

## Dependency

`umap-learn` joins the base `dependencies` in `backend/pyproject.toml`. It brings in numba, llvmlite and pynndescent, about 60 MB, and reaches the API, default and gpu images and CI with no Dockerfile edits. `uv.lock` is regenerated.

## Frontend

```
frontend/src/shared/components/chemical-space/
  view.ts            pure: fit-to-bounds, zoom about a point, pan, map<->screen, picking grid
  renderer.ts        WebGL2 point renderer (port of ChemCellar's), styles and view uniforms
  chemical-space-map.tsx  React host: canvas, resize, pointer, wheel, double-click reset, hover, fallback
  *.test.ts(x)
features/runs/components/run-chemical-space.tsx        run page card
features/protocols/components/protocol-chemical-space.tsx  protocol page card
```

- **Point styles**, colored from tokens through uniforms, so a theme switch re-reads colors without re-uploading points:

  | Style | Look | Token |
  |---|---|---|
  | Training | 2.5 px dot, alpha 0.45 | `chart-1` |
  | Validation | 2.5 px dot | `chart-2` |
  | Test | 3 px dot | `ds-score-fair` |
  | Run, in domain (similarity ≥ 0.3) | 7 px filled disk with a card-colored edge | `ds-score-fair` |
  | Run, out of domain | 8 px dashed ring, drawn in the fragment shader by angle | `ds-score-fair` |
  | Highlighted neighbor | 6 px ring | `foreground` |

- **Draw order:** training cloud, then run points on top. Two draw calls.
- **Interaction:** wheel zooms about the cursor, drag pans, double-click resets.
- **Hover:** picking through a uniform grid over run points (run page) or all points (protocol page). Hovering opens a tooltip with `StructureThumbnail`, the id, the prediction and the real similarity, and draws lines to the 5 neighbors (run page).
- **Entrance:** the training cloud fades in and run points scale in, once. None under reduced motion.
- **Fallback:** without WebGL2, the card shows the text summary and "This browser cannot draw the map." No crash.
- **Accessibility:** the canvas has `role="img"` and an `aria-label` with the real counts. The summary is plain text above the map.
- **Run page:** "Chemical space" card above the triage grid. It replaces the illustrative applicability explainer there; that explainer stays on the scorecard diagnostics.
  - Summary: "{in} of {total} compounds are inside the applicability domain (similarity ≥ 0.3). The nearest training compound is at similarity {min} to {max}."
  - Caption: "Each dot is a compound, placed by UMAP so that similar structures sit together. Position shows which part of the training set a compound is closest to; distance on the map is approximate. A filled marker is inside the applicability domain; a dashed ring is outside it."
- **Protocol page:** "Chemical space" card in the scorecard, before "Scaffold split versus random split". Training is blue, validation teal, test amber, with counts and the split strategy named.
- **`"missing"` state:** "No map has been computed for this protocol yet."

## Testing

- **Backend unit tests:**
  - `nearest_neighbours_tanimoto` matches a brute-force reference on random fingerprints, including ties and `k` larger than the reference set, and the `k = 1` path equals the old function's output.
  - `umap_layout` is deterministic for a seed, stays inside [0, 1], and puts two clearly separated scaffold series in separate regions.
  - Placement is a weighted mean, with an unweighted fallback when all similarities are 0.
- **Backend API tests:**
  - Training a protocol produces a ready map whose train order matches `train_structures`.
  - A protocol with no map returns `missing`.
  - A prediction run returns placed points whose neighbors index train points.
  - The compounds endpoints cap at 50.
  - The backfill builds a map for a protocol without one, and re-running it is a no-op.
- **Frontend unit tests:** `view.ts` (zooming about a point keeps that point fixed, fit-to-bounds, nearest-pick within radius); the map host falls back without WebGL; the run card summary text from fixture data.
- **Live check:** restart the workers (they do not hot-reload), run the backfill, and open the PAINS run and its protocol. Benchmark `umap_layout` on 100,000 synthetic sparse fingerprints and draw 100,000 points in the browser; record both timings.

## Out of scope

Lasso selection to collections, coloring by predicted value, a per-protocol map on the dataset page, and binary payloads (Caddy already compresses with zstd/gzip). Each is a follow-up once the map exists.

## Risks

- **Map fitting time, measured** (seeded UMAP, synthetic sparse ECFP4-like bits, M-series Mac): 10.4 s at 5,000 (mostly numba's one-time compile), 36.1 s at 50,000, 71.5 s and 1.1 GB peak at 100,000. RDKit fingerprinting adds roughly 15–30 s per 100,000. The 150,000 fit ceiling keeps the worst case near 2 minutes and 1.7 GB.
- **Payload size** of about 2 MB of JSON per view at 100,000, compressed in production. Binary is the upgrade path.
- **A run of novel compounds always lands inside the cloud.** This is inherent to neighbor placement. The ring style, the summary and the hover similarity carry the truth, and the caption says so.
