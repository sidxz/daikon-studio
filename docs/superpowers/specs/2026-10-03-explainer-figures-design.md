# Explainer figures and navigation icons

Date: 2026-10-03 · Branch: `explainer-figures` · Status: design approved through preview; awaiting spec review

## Goal

Studio's audience includes biologists, medicinal chemists, program leadership, and scientists new to machine learning. Several screens ask them to judge ML concepts they may never have seen drawn: what an engine does, why a split choice changes the score, what a bootstrap interval means, and why a far-away compound deserves less trust. This work adds small figures that show those mechanisms with plain computer-science examples (points, trees, curves, graphs, tokens) rather than chemistry, plus a custom set of navigation icons.

It must not make the app feel busier. Every figure explains one concept on the screen where that concept is decided, sits behind a disclosure, and animates once.

## Decisions already made

| Decision | Choice | Source |
|---|---|---|
| Placements | A (engines), B (splits), C (bootstrap), D (applicability domain), G (nav icons). Dashboard pipeline and empty states are out of scope | Placement question |
| Visibility | "How this works" disclosure. Open on the first visit, then remembered per browser and per figure | Visibility question |
| Rendering | Inline SVG, React, no new dependency, no WebGL | Rendering question |
| Motion | Plays once when 35% of the figure is in view, replays on hover or Replay, final frame only under `prefers-reduced-motion` | Motion question |
| Figures A1–A5, B, C, D | Keep as previewed | Preview review, all marked Keep |
| Icons G | Revise: "make them colored" | Preview review |
| Icon color treatment | Colored glyphs: each icon in its own hue, labels neutral | Preview G2, option 1 |
| Logo | Unchanged | Brief |

The approved preview is https://claude.ai/artifact/6P4w9QLvvhEFTjfJUHEEEe, and its source is committed beside this spec as `2026-10-03-explainer-figures-preview.html`. That file is the reference for geometry, timing, and copy; this spec does not repeat coordinates.

## Color key

All figures share one key, taken from the existing tokens:

| Meaning | Token | Tailwind |
|---|---|---|
| Training data, measured values | `--chart-1` | `fill-chart-1` / `stroke-chart-1` |
| Test or new inputs, residuals | `--ds-score-fair` (amber, darker than `--chart-4` in light mode for contrast) | `fill-score-fair` |
| Messages in A4 | `--chart-2` | `fill-chart-2` |
| Model output | `--foreground`, drawn as an open ring | `stroke-foreground` |
| Structure (edges, axes, boxes) | `--border`, `--muted-foreground` | existing |

"Open ring = predicted, filled dot = measured" mirrors the logo's measured/predicted halves. D uses a dashed ring for an out-of-domain prediction.

## Architecture

```
frontend/src/shared/components/explainers/
  use-timeline.ts          play-once/replay/reduced-motion clock -> t in [0, 1]
  explainer.tsx            disclosure + figure frame + caption + Replay button
  explainer-store.ts       zustand persist: { [figureId]: open }
  math/
    prng.ts                mulberry32 + Box-Muller (seeded)
    boosting.ts            stump boosting, returns per-round step functions
    gaussian-process.ts    RBF posterior via Cholesky, mean and sd on a grid
    bootstrap.ts           resample accuracies + percentile interval
    nearest.ts             nearest-neighbor links and median similarity
    *.test.ts
  figures/
    forest.tsx  boosting.tsx  gaussian-process.tsx  message-passing.tsx  attention.tsx
    split.tsx   bootstrap.tsx  domain.tsx
frontend/src/shared/components/icons/nav-icons.tsx
```

### `useTimeline(ref, durationMs)`

Returns `{ t, replay }`. It starts at `t = 1` so server rendering, thumbnails, and reduced motion all show the complete final frame. The first time an IntersectionObserver (threshold 0.35) sees the element, it runs `requestAnimationFrame` from 0 to 1, then stops. `pointerenter` and `replay()` restart it if it is idle. Under `prefers-reduced-motion: reduce` it stays at 1. A frame loop runs only while a figure is playing.

Each figure is a pure function of `t`: `(t, data) => <svg>`. Data is computed once with `useMemo` from fixed seeds, so the drawing is deterministic and the math is testable without a DOM. Re-rendering ~110 SVG nodes per frame for a few seconds is well within React's budget; if profiling ever says otherwise, the upgrade is setting attributes through refs, not a library.

### `<Explainer id label caption>{figure}</Explainer>`

Built on the existing `Collapsible` (Radix). The trigger is a text button ("How this works" or "How it learns") with a chevron. The open state comes from `explainer-store`, which defaults to open when an id has no stored value, and persists under the key `ds-explainers` with the same `persist` pattern as `font-scale-store`. Opening a closed explainer replays its figure. The frame holds the SVG (`role="img"`, `aria-label` summarizing what the final frame shows), the caption, and a Replay button with a visible focus state.

Colors come from Tailwind token classes on SVG elements, so theme switches need no JavaScript. `useChartTheme` is not needed, because nothing interpolates colors.

## Placements

| Id | Figure | Where | File |
|---|---|---|---|
| A1 | Random forest: four trees route one input, votes average to a prediction, spread shown | Engine card on the Engines page; under the Engine select in Train a protocol | `engine-catalogue.tsx`, `train-protocol-form.tsx` |
| A2 | Gradient boosting: six rounds of stumps fit residuals | same, for `ecfp4-xgboost`, `ecfp4-lightgbm`, `descriptors-xgboost` | same |
| A3 | Gaussian process: band narrows near five observations | same, for `tanimoto-gp` | same |
| A4 | Message passing: three rounds, receptive field grows, sum to a learned vector | same, for `chemprop-dmpnn` | same |
| A5 | Attention: arcs between seven tokens, pretrained encoder, trained output layer | same, for `molformer-xl` | same |
| B | Random vs. scaffold split, nearest-training links, median-similarity bars | Dataset wizard step "Split", driven by the selected option; dataset page next to the train–test similarity card, fixed to the dataset's own strategy | `dataset-wizard.tsx`, `dataset-profile-view.tsx` |
| C | Bootstrap redraws build the 95% interval; baseline inside or outside | Scorecard verdict, under the interval line, only when `verdict.ci` exists. The example opens on the outcome that matches the real verdict (`beats` shows the baseline outside the interval; anything else shows it inside) and keeps the two-option toggle | `scorecard-view.tsx` |
| D | New input leaves the training cloud; error rises; dashed ring outside similarity 0.3 | Scorecard diagnostics, distance-vs-error card; above the triage grid as "How to read uncertainty and applicability" | `scorecard-diagnostics.tsx`, `run-detail.tsx` |

Engine-to-figure mapping lives beside the engine types as `ENGINE_FIGURE: Record<string, FigureKind>`. An engine id with no entry shows no explainer, so a newly added engine never shows the wrong picture.

**Simplification from the preview brief:** D in the triage view opens from a disclosure above the grid, not from the AG Grid column header. A custom header component with a popover adds a grid-specific surface for no explanatory gain. If you want it in the header, say so.

D's threshold uses the existing constants (`IN_DOMAIN_FLOOR = 0.3` in the frontend; `_WITHIN_DOMAIN_THRESHOLD = 0.3` in the backend), so the figure and the triage grid's warning color agree.

## Icons (G)

Seven custom icons on lucide's 24-unit grid with a 2-unit round stroke, so they sit beside the lucide icons that stay on buttons. Dashboard keeps lucide's `LayoutDashboard`.

| Item | Glyph | Hue (light / dark) |
|---|---|---|
| Datasets | Table, filled dots per row (measured) | blue `--chart-1` |
| Protocols | Hexagon with a check (validated, published) | green `--ds-success` |
| Sweeps | One filled dot fanned to three rings | violet `--chart-3` |
| Runs | Play triangle, two open rings (predictions) | amber, new `--icon-runs` (#b45309 / #fbbf24) |
| Collections | Dashed selection around three dots | pink, new `--icon-collections` (#db2777 / #f472b6) |
| Engines | Axes, a fitted curve, three dots | teal `--ds-score-good` |
| Runners | Two slabs with status lights | cyan, new `--icon-runners` (#0891b2 / #22d3ee) |

Each icon is a component accepting `className`, with `currentColor` strokes. `NavItem.icon` widens from `LucideIcon` to `ComponentType<{ className?: string }>`; its three consumers (`nav-main.tsx`, `command-palette.tsx`, the dashboard page) need no other change. The hue is applied with `text-*` utility classes in `navigation.ts`, so the sidebar, palette, and dashboard tiles all pick it up. The three hues the token package lacks are declared in `globals.css` for both themes. Measured against `--sidebar` (#f1f5f9) in light mode, every hue reaches at least 3.0:1, including on the active row (#dfe9f8). The token amber (`--ds-score-fair`, 2.9:1) fails there, which is why Runs gets its own darker amber. In dark mode every hue is above 6.5:1.

Known weak spots, accepted for now: Sweeps can read as a share icon, and Engines as a line chart.

## Copy

Captions are the preview's captions, which already follow `docs/copy-audit.md`: American spelling, sentence case, no em-dash chains, noun-phrase headings. Each caption states where the figure simplifies: one dimension instead of Tanimoto space, accuracy instead of the primary metric, 200 of 1,000 redraws, and an illustrative error curve. That honesty is part of the design and must not be edited away.

## Accessibility

- Every SVG has `role="img"` and an `aria-label` describing the final frame; decorative sub-elements are not exposed.
- Replay is a real button. The disclosure is Radix Collapsible, so `aria-expanded` and keyboard handling come with it.
- Reduced motion shows the final frame and never runs a frame loop.
- Meaning never depends on color alone. Test vs. training is also shown by the link lines; predicted vs. measured by open vs. filled; out-of-domain by a dashed ring.

## Testing

- **Unit (vitest)**: each `math/*` module, with seeded checks that fail if the logic breaks. Boosting: mean error decreases every round. GP: posterior sd at an observation is below 0.05 and at the widest gap above 0.15. Bootstrap: 95% of resamples fall inside the interval, and with seed 21 the interval is [0.625, 0.900]. Nearest: scaffold median similarity is below random median similarity.
- **`useTimeline`**: with fake timers and a mocked IntersectionObserver, it rests at 1, plays to 1 once in view, does not replay on a second intersection, and never leaves 1 under reduced motion.
- **Components**: one smoke render per figure at `t = 1`, asserting its `aria-label` and that no `NaN` reaches an SVG attribute.
- **Live check**: `make dev`, signed in with a real account (no test bypass exists), with every placement checked in both themes and with reduced motion. A screenshot per placement goes into the PR.

## Out of scope

Dashboard pipeline figure, empty-state illustrations, training-progress animation (the backend reports no progress to animate), WebGL, any change to the logo, and per-user server-side persistence of disclosure state.

## Risks

- **Drift between the figures and engine behavior.** If an engine's uncertainty definition changes (for example, chemprop gains MVE uncertainty), the A-figure captions go stale. The `ENGINE_FIGURE` map and the captions sit next to the engine types, so a reviewer of an engine change sees them.
- **Bundle size.** About eight small components plus math, all client-only and loaded with the pages that use them. No new dependency.
