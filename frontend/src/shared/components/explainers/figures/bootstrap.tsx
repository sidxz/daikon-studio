"use client";

import { cn } from "@/shared/lib/utils";
import { useState } from "react";
import { Explainer } from "../explainer";
import { bootstrapExample } from "../math/bootstrap";
import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const BOOTSTRAP_MS = 9800;
export const BOOTSTRAP_CAPTION =
  "The test set is redrawn many times with replacement, so some examples appear twice and others not at all. Each redraw is scored, and the middle 95% of those scores is the interval. Shown with accuracy on 200 redraws; Studio uses the protocol's primary metric and 1,000 redraws.";

const EX = bootstrapExample();
const N = EX.correct.length;
const B = EX.accuracies.length;
const [LO, HI] = EX.interval;
const AXIS_Y = 262;
const STEP = 4.6;
const DEMO = 3;
const DEMO_W = 0.11;
const DEMO_0 = 0.06;
const SX = (i: number) => 60 + i * 13;
const X = (a: number) => 60 + ((a - 0.5) / 0.5) * 516;
/** Each redraw's dot, stacked by accuracy (accuracies are exact multiples of 1/40). */
const DOTS: [number, number][] = (() => {
  const counts = new Map<number, number>();
  return EX.accuracies.map((a) => {
    const c = (counts.get(a) ?? 0) + 1;
    counts.set(a, c);
    return [X(a), AXIS_Y - 4 - (c - 1) * STEP];
  });
})();

/** The two example baselines: one inside the interval, one below it. */
const INSIDE = 0.7;
const OUTSIDE = 0.55;

export function BootstrapFigure({ t, baseline }: { t: number; baseline: number }) {
  let demo = -1;
  for (let d = 0; d < DEMO; d++)
    if (t >= DEMO_0 + d * DEMO_W && t < DEMO_0 + (d + 1) * DEMO_W) demo = d;
  const picks = new Array<number>(N).fill(0);
  let status = t >= DEMO_0 + DEMO * DEMO_W ? `${B} redraws` : "";
  if (demo >= 0) {
    const s0 = DEMO_0 + demo * DEMO_W;
    const shown = Math.floor(seg(t, s0, s0 + DEMO_W * 0.6) * N);
    for (let k = 0; k < shown; k++) picks[EX.draws[demo][k]]++;
    status =
      t >= s0 + DEMO_W * 0.6
        ? `redraw ${demo + 1}: accuracy ${EX.accuracies[demo].toFixed(3)}`
        : `redraw ${demo + 1}: drawing 40 with replacement`;
  }
  const inside = baseline >= LO && baseline <= HI;

  return (
    <div className="space-y-2">
      <svg
        viewBox="0 0 640 300"
        className="block h-auto w-full overflow-visible"
        role="img"
        aria-label="A test set of 40 examples, 30 predicted correctly. It is resampled with replacement 200 times; each resample's accuracy is stacked as a dot. The middle 95% of dots forms the interval, and the baseline is drawn as a dashed line."
      >
        <text x={60} y={12} className={S.textStrong}>
          Test set: 40 examples, 30 predicted correctly
        </text>
        <text x={576} y={12} textAnchor="end" className={S.text}>
          {status}
        </text>
        {EX.correct.map((ok, i) => {
          const a = (i / N) * 0.05;
          return (
            <rect
              // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length, never reordered
              key={i}
              x={SX(i)}
              y={34}
              width={9}
              height={9}
              rx={1.5}
              opacity={seg(t, a, a + 0.02)}
              className={ok ? S.measured : S.heldRing}
              strokeWidth={ok ? 0 : 1.5}
            />
          );
        })}
        <g opacity={demo >= 0 ? 1 : 0}>
          {picks.map((count, i) =>
            [0, 1, 2].map((j) => (
              <circle
                key={`${i}-${j}`}
                cx={SX(i) + 4.5}
                cy={28 - j * 4}
                r={1.5}
                opacity={j < count ? 1 : 0}
                className={S.inkFill}
              />
            )),
          )}
        </g>
        <g opacity={seg(t, 0.8, 0.88)}>
          <rect x={X(LO)} y={88} width={X(HI) - X(LO)} height={AXIS_Y - 88} className={S.band} />
          <line
            x1={X(LO)}
            y1={88}
            x2={X(LO)}
            y2={AXIS_Y}
            className={S.measuredLine}
            strokeWidth={1}
          />
          <line
            x1={X(HI)}
            y1={88}
            x2={X(HI)}
            y2={AXIS_Y}
            className={S.measuredLine}
            strokeWidth={1}
          />
          <text x={X(HI) - 6} y={102} textAnchor="end" className={S.textStrong}>
            95% interval {LO.toFixed(3)} to {HI.toFixed(3)}
          </text>
        </g>
        <line x1={60} y1={AXIS_Y} x2={576} y2={AXIS_Y} className={S.line} />
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((v) => (
          <g key={v}>
            <line x1={X(v)} y1={AXIS_Y} x2={X(v)} y2={AXIS_Y + 4} className={S.line} />
            <text x={X(v)} y={AXIS_Y + 16} textAnchor="middle" className={S.text}>
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={576} y={AXIS_Y + 32} textAnchor="end" className={S.text}>
          accuracy on each redraw
        </text>
        {DOTS.map(([x, y], b) => {
          let a: number;
          if (b < DEMO)
            a = ease(seg(t, DEMO_0 + b * DEMO_W + DEMO_W * 0.7, DEMO_0 + (b + 1) * DEMO_W));
          else {
            const s = 0.4 + ((b - DEMO) / (B - DEMO)) * 0.38;
            a = ease(seg(t, s, s + 0.02));
          }
          return (
            <circle
              // biome-ignore lint/suspicious/noArrayIndexKey: one dot per redraw, fixed order
              key={b}
              cx={b < DEMO ? lerp(318, x, a) : x}
              cy={b < DEMO ? lerp(52, y, a) : y - (1 - a) * 10}
              r={2.2}
              opacity={a > 0 ? 1 : 0}
              className={S.measured}
            />
          );
        })}
        <g opacity={seg(t, 0.88, 0.94)}>
          <line
            x1={X(baseline)}
            x2={X(baseline)}
            y1={64}
            y2={AXIS_Y}
            className={S.ink}
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <text x={X(baseline)} y={58} textAnchor="middle" className={S.textStrong}>
            baseline {baseline.toFixed(2)}
          </text>
        </g>
      </svg>
      <p className="flex flex-wrap items-center gap-2 text-sm" style={{ opacity: seg(t, 0.93, 1) }}>
        <span
          className={cn(
            "rounded-md px-2 py-0.5 text-xs font-medium",
            inside ? "bg-warning/10 text-warning" : "bg-success/10 text-success",
          )}
        >
          {inside ? "Within noise" : "Beats the baseline"}
        </span>
        <span>
          {inside
            ? `The baseline's ${baseline.toFixed(2)} lies inside the interval, so the two models are not distinguishable on this test set.`
            : `The baseline's ${baseline.toFixed(2)} lies below the interval, so the model's advantage is larger than the test set's sampling noise.`}
        </span>
      </p>
    </div>
  );
}

/** C on the scorecard: opens on the outcome that matches the real verdict, with both examples one click away. */
export function BootstrapExplainer({ startOutside }: { startOutside: boolean }) {
  const [baseline, setBaseline] = useState(startOutside ? OUTSIDE : INSIDE);
  return (
    <Explainer
      id="bootstrap"
      label="How the interval is computed"
      durationMs={BOOTSTRAP_MS}
      caption={BOOTSTRAP_CAPTION}
    >
      {(t) => (
        <div className="space-y-2">
          <fieldset className="inline-flex gap-0.5 rounded-lg border bg-card p-0.5">
            <legend className="sr-only">Example baseline</legend>
            {[INSIDE, OUTSIDE].map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={baseline === value}
                onClick={() => setBaseline(value)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-xs",
                  baseline === value
                    ? "bg-accent font-medium text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                Baseline {value.toFixed(2)}
              </button>
            ))}
          </fieldset>
          <BootstrapFigure t={t} baseline={baseline} />
        </div>
      )}
    </Explainer>
  );
}
