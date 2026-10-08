"use client";

import { useMemo } from "react";
import { Explainer } from "../explainer";
import { type BootstrapData, bootstrapLayout } from "../math/bootstrap";
import { mulberry32 } from "../math/prng";
import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export type { BootstrapData } from "../math/bootstrap";

export const BOOTSTRAP_MS = 9800;

const AXIS_Y = 262;
const STEP = 4.6;
const DEMO = 3;
const DEMO_W = 0.11;
const DEMO_0 = 0.06;
const SX = (i: number) => 60 + i * 13;
/** Which shown compound each pick of the three demonstration redraws lands on. */
const DEMO_PICKS = (() => {
  const rand = mulberry32(21);
  return Array.from({ length: DEMO }, () => Array.from({ length: 40 }, () => rand()));
})();

const fmt = (v: number) => v.toFixed(3);

export function BootstrapFigure({ t, data }: { t: number; data: BootstrapData }) {
  const L = useMemo(() => bootstrapLayout(data), [data]);
  const { metric, baseline, testSize: n } = data;
  const [lo, hi] = data.interval;
  const N = L.cells.length;
  const X = (v: number) => 60 + ((v - L.domain[0]) / (L.domain[1] - L.domain[0])) * 516;
  const binX = (bin: number) => X((data.redraws.edges[bin] + data.redraws.edges[bin + 1]) / 2);
  const from = SX(N / 2);

  let demo = -1;
  for (let d = 0; d < DEMO; d++)
    if (t >= DEMO_0 + d * DEMO_W && t < DEMO_0 + (d + 1) * DEMO_W) demo = d;
  const picks = new Array<number>(N).fill(0);
  let status = t >= DEMO_0 + DEMO * DEMO_W ? `${L.total.toLocaleString()} reshuffled tests` : "";
  if (demo >= 0) {
    const s0 = DEMO_0 + demo * DEMO_W;
    const shown = Math.floor(seg(t, s0, s0 + DEMO_W * 0.6) * N);
    for (let k = 0; k < shown; k++) picks[Math.floor(DEMO_PICKS[demo][k] * N)]++;
    status =
      t >= s0 + DEMO_W * 0.6
        ? `Reshuffled test ${demo + 1}: scored`
        : `Reshuffled test ${demo + 1}: picking ${n.toLocaleString()} compounds at random`;
  }

  const inside = baseline >= lo && baseline <= hi;
  const modelAhead = data.higherIsBetter ? baseline < lo : baseline > hi;
  const diff = data.mode === "difference";
  const reference = diff ? "no difference" : `baseline model: ${fmt(baseline)}`;
  const better = data.higherIsBetter ? "higher" : "lower";

  return (
    <div className="space-y-2">
      <svg
        viewBox="0 0 640 300"
        className="block h-auto w-full overflow-visible"
        role="img"
        aria-label={
          diff
            ? `Model minus baseline ${metric} on ${L.total.toLocaleString()} reshuffled tests, shown as a pile of dots. The likely range of the difference, ${fmt(lo)} to ${fmt(hi)}, is shaded. Zero, where the two models tie, is marked against it.`
            : `The model's ${metric} on ${L.total.toLocaleString()} reshuffled tests, shown as a pile of dots. The likely range of the model's score, ${fmt(lo)} to ${fmt(hi)}, is shaded. The baseline model's ${fmt(baseline)} is marked against it.`
        }
      >
        <text x={60} y={12} className={S.textStrong}>
          {N < n ? `${N} of the` : "All"} {n.toLocaleString()} test compounds
        </text>
        <text x={60} y={56} opacity={seg(t, 0, 0.05)} className={S.small}>
          {data.cutoff != null
            ? "filled = predicted right, open = predicted wrong"
            : "bigger square = bigger error"}
        </text>
        <text x={576} y={12} textAnchor="end" className={S.text}>
          {status}
        </text>
        {L.cells.map((cell, i) => {
          const a = (i / N) * 0.05;
          // Classification: filled if predicted correctly. Regression: area by error.
          const w = data.cutoff != null ? 9 : 2 + 7 * cell.size;
          const off = (9 - w) / 2;
          const filled = data.cutoff == null || cell.ok;
          return (
            <rect
              // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length, never reordered
              key={i}
              x={SX(i) + off}
              y={34 + off}
              width={w}
              height={w}
              rx={1.5}
              opacity={seg(t, a, a + 0.02)}
              className={filled ? S.measured : S.heldRing}
              strokeWidth={filled ? 0 : 1.5}
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
          <rect x={X(lo)} y={88} width={X(hi) - X(lo)} height={AXIS_Y - 88} className={S.band} />
          <line x1={X(lo)} y1={88} x2={X(lo)} y2={AXIS_Y} className={S.measuredLine} />
          <line x1={X(hi)} y1={88} x2={X(hi)} y2={AXIS_Y} className={S.measuredLine} />
        </g>
        <line x1={60} y1={AXIS_Y} x2={576} y2={AXIS_Y} className={S.line} />
        {L.ticks.map((v) => (
          <g key={v}>
            <line x1={X(v)} y1={AXIS_Y} x2={X(v)} y2={AXIS_Y + 4} className={S.line} />
            <text x={X(v)} y={AXIS_Y + 16} textAnchor="middle" className={S.text}>
              {v}
            </text>
          </g>
        ))}
        <text x={576} y={AXIS_Y + 32} textAnchor="end" className={S.text}>
          {diff
            ? `${metric}, model minus baseline (${better} favors the model)`
            : `${metric} (${better} is better)`}
        </text>
        {L.dots.map(({ bin, level }, b) => {
          const x = binX(bin);
          const y = AXIS_Y - 4 - level * STEP;
          let a: number;
          if (b < DEMO)
            a = ease(seg(t, DEMO_0 + b * DEMO_W + DEMO_W * 0.7, DEMO_0 + (b + 1) * DEMO_W));
          else {
            const s = 0.4 + ((b - DEMO) / Math.max(1, L.dots.length - DEMO)) * 0.38;
            a = ease(seg(t, s, s + 0.02));
          }
          return (
            <circle
              // biome-ignore lint/suspicious/noArrayIndexKey: one dot per arrival, fixed order
              key={b}
              cx={b < DEMO ? lerp(from, x, a) : x}
              cy={b < DEMO ? lerp(52, y, a) : y - (1 - a) * 10}
              r={2.2}
              opacity={a > 0 ? 1 : 0}
              className={S.measured}
            />
          );
        })}
        <g opacity={seg(t, 0.88, 0.94)}>
          {L.baselineOff === 0 ? (
            <>
              <line
                x1={X(baseline)}
                x2={X(baseline)}
                y1={84}
                y2={AXIS_Y}
                className={S.ink}
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
              <text x={X(baseline)} y={78} textAnchor="middle" className={S.textStrong}>
                {reference}
              </text>
            </>
          ) : (
            // Too far to share the axis without crushing the dots: named at the edge.
            <text
              x={L.baselineOff < 0 ? 60 : 576}
              y={78}
              textAnchor={L.baselineOff < 0 ? "start" : "end"}
              className={S.textStrong}
            >
              {L.baselineOff < 0 ? `← ${reference}` : `${reference} →`}
            </text>
          )}
        </g>
        <text textAnchor="end" opacity={seg(t, 0.8, 0.88)} className={S.textStrong}>
          <tspan x={Math.max(X(hi) - 6, 266)} y={102}>
            {diff ? "likely range of the difference" : "likely range of the model's score"}
          </tspan>
          <tspan x={Math.max(X(hi) - 6, 266)} y={115}>
            {fmt(lo)} to {fmt(hi)} (95% interval)
          </tspan>
        </text>
      </svg>
      <p className="text-sm" style={{ opacity: seg(t, 0.93, 1) }}>
        {diff
          ? inside
            ? "Zero is inside the likely range of the difference, so this test can't tell which model is better."
            : `Every likely difference favors the ${modelAhead ? "model" : "baseline"}, so its lead holds up when the test compounds are reshuffled.`
          : inside
            ? `The baseline (${fmt(baseline)}) is inside the model's likely range, so this test can't tell which model is better.`
            : modelAhead
              ? `The baseline (${fmt(baseline)}) is outside the model's likely range, so the model is genuinely better, not just lucky with the test compounds.`
              : `The baseline (${fmt(baseline)}) is outside the model's likely range, so the baseline is genuinely better, not just lucky with the test compounds.`}
      </p>
    </div>
  );
}

/** C on the scorecard, drawn from this card's own test set and baseline. */
export function BootstrapExplainer({ data }: { data: BootstrapData }) {
  const total = data.redraws.counts.reduce((a, b) => a + b, 0).toLocaleString();
  return (
    <Explainer
      id="bootstrap"
      label="What the 95% interval means"
      durationMs={BOOTSTRAP_MS}
      caption={
        data.mode === "difference"
          ? `A model's score depends partly on which compounds happened to be in the test set. To see how much, Studio makes ${total} reshuffled tests. Each one picks compounds at random from the real test set, so some appear twice and others not at all, and scores both models on the same picks. The pile of dots shows how far apart the two scores were each time. If zero falls outside the shaded range, one model stays ahead however the test compounds are reshuffled.`
          : `A model's score depends partly on which compounds happened to be in the test set. To see how much, Studio makes ${total} reshuffled tests. Each one picks compounds at random from the real test set, so some appear twice and others not at all. The pile of dots shows the model's score on each. If the baseline falls outside the shaded range, the difference between the two models is real.`
      }
    >
      {(t) => <BootstrapFigure t={t} data={data} />}
    </Explainer>
  );
}
