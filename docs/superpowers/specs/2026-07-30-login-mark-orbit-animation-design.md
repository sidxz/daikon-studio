# Login mark — spectrum orbit animation

**Date:** 2026-07-30
**Status:** Approved direction (mockups: claude.ai artifact "spectrum orbits, refinements")

## Context

The login page's cover watermark (the hexagon brand mark) sits at 7% opacity and is
nearly invisible in light mode, and the page has no motion. Sid wants the mark animated
with a modern orb-like quality — subtle, meaningful, and confined to the logo itself:
clean background, no glow orbs behind the mark, geometry and the solid/dashed halves
untouched. Direction was chosen from animated mockups after two reference videos
(gradient-orb aesthetics); the winning study is "the spectrum orbits."

## Decision

- **Animation:** the brand gradient rotates around the mark's center, so the suite
  spectrum circulates through the strokes. Constant velocity, linear, endless.
- **Speed:** 7 seconds per full lap.
- **Legibility fix (separate from motion):** watermark opacity 0.07 → **0.28 light /
  0.20 dark**, plus a soft same-hue bloom (`drop-shadow`, ~12px at rendered size,
  rgba(96, 130, 255, 0.20)).
- **Scope:** only the login cover watermark animates. The small header logo (and every
  other LogoMark use) stays static.
- **Reduced motion:** users with `prefers-reduced-motion` get the static mark with the
  same darkened opacity and bloom.

## Implementation

Two files change; no new dependencies (honors the page's existing "no animation library"
rule).

### `frontend/src/shared/components/logo-mark.tsx`

Add an optional `animate` prop (default `false`). When set, the gradient def includes
one SMIL element:

```svg
<animateTransform attributeName="gradientTransform" type="rotate"
  from="0 16 16" to="360 16 16" dur="7s" repeatCount="indefinite"/>
```

Nothing else in the component changes; `useId` already prevents gradient id collisions
across instances. The 7s duration is hardcoded — it is the brand clock, not a knob.

### `frontend/src/app/login/page.tsx`

The cover renders two watermark instances, CSS-swapped so reduced-motion needs no JS:

- `<LogoMark animate className="… motion-safe:block motion-reduce:hidden" />`
- `<LogoMark className="… motion-reduce:block motion-safe:hidden" />`

Shared watermark classes change from `opacity-[0.07]` to `opacity-[0.28]
dark:opacity-[0.20]` plus the bloom filter (arbitrary Tailwind
`[filter:drop-shadow(0_0_12px_rgba(96,130,255,0.2))]`). The stale "no animation" comment
is updated to describe the SMIL approach.

## Out of scope

The other mockup studies (liquid glass fill, wandering inner light, breath/dash-tick,
tide/surge/stream cadences) are documented in the artifact and not implemented. The
dash-tick on the predicted half composes with this design if ever wanted.

## Verification

- `pnpm dev`, visit `/login`: gradient visibly orbits at 7s/lap; mark clearly legible on
  white; check dark theme; toggle OS reduce-motion → static mark, same legibility.
- Existing e2e auth flow unaffected (page structure and controls unchanged).
