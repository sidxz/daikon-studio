import { useMemo } from "react";
import { domainExample, nearestPoint, typicalError } from "../math/domain";
import { easeInOut, seg } from "../math/tween";
import { S } from "./styles";

export const DOMAIN_MS = 7800;

export function domainCaption(where: "diagnostics" | "triage", threshold: number): string {
  const lead = `A model is most reliable near the examples it learned from. As a new input moves away, its similarity to the nearest training point falls and the typical error rises. Below a similarity of ${threshold}, the input is outside the applicability domain and its prediction is drawn as an open, dashed ring. The curve is illustrative;`;
  return where === "diagnostics"
    ? `${lead} the chart below this panel shows your protocol's measured version.`
    : `${lead} the scorecard's diagnostics show your protocol's measured version.`;
}

const CX0 = 384;
const CX1 = 624;
const CY0 = 30;
const CY1 = 226;
const CX = (s: number) => CX0 + s * (CX1 - CX0);
const CY = (e: number) => CY1 - e * (CY1 - CY0);
const CURVE = Array.from({ length: 61 }, (_, i) => {
  const s = i / 60;
  return `${i ? "L" : "M"}${CX(s)} ${CY(typicalError(s))}`;
}).join(" ");

function Marker({
  cx,
  cy,
  r,
  out,
  width,
}: { cx: number; cy: number; r: number; out: boolean; width: number }) {
  return (
    <circle
      cx={cx}
      cy={cy}
      r={r}
      className={out ? S.heldRing : S.held}
      strokeWidth={width}
      strokeDasharray={out ? "2.4 2.1" : undefined}
    />
  );
}

export function DomainFigure({ t, threshold }: { t: number; threshold: number }) {
  const ex = useMemo(() => domainExample(threshold), [threshold]);
  const s = easeInOut(seg(t, 0.18, 0.86));
  const pt = ex.path[Math.round(s * (ex.path.length - 1))];
  const near = ex.training[nearestPoint(ex.training, pt.x, pt.y).index];
  const out = pt.similarity < threshold;
  const moving = t > 0.18 && t < 0.9 ? 1 : 0;
  const bar = 5 + 26 * typicalError(pt.similarity);

  return (
    <svg
      viewBox="0 0 640 270"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="A new input moves away from the training data through three positions. Its similarity to the nearest training point falls, its error bar widens, and at position 3 it leaves the applicability domain."
    >
      <g opacity={0.09 * seg(t, 0.08, 0.16)}>
        {ex.training.map(([x, y]) => (
          <circle key={`d${x}:${y}`} cx={x} cy={y} r={ex.reach} className={S.measured} />
        ))}
      </g>
      <text x={16} y={18} opacity={seg(t, 0.1, 0.16)} className={S.text}>
        applicability domain (similarity ≥ {threshold})
      </text>
      {ex.training.map(([x, y], i) => {
        const a = (i / ex.training.length) * 0.1;
        return (
          <circle
            key={`p${x}:${y}`}
            cx={x}
            cy={y}
            r={2.8}
            opacity={seg(t, a, a + 0.03)}
            className={S.measured}
          />
        );
      })}
      <rect
        x={CX0}
        y={CY0}
        width={CX(threshold) - CX0}
        height={CY1 - CY0}
        className="fill-score-fair/10"
      />
      <line
        x1={CX(threshold)}
        y1={CY0}
        x2={CX(threshold)}
        y2={CY1}
        className={S.ink}
        strokeWidth={1}
        strokeDasharray="3 3"
      />
      <text x={CX0 + 4} y={CY1 - 19} className={S.small}>
        outside
      </text>
      <text x={CX0 + 4} y={CY1 - 8} className={S.small}>
        domain
      </text>
      <line x1={CX0} y1={CY1} x2={CX1} y2={CY1} className={S.line} />
      <line x1={CX0} y1={CY0} x2={CX0} y2={CY1} className={S.line} />
      {[0, threshold, 1].map((v) => (
        <text key={v} x={CX(v)} y={CY1 + 14} textAnchor="middle" className={S.text}>
          {v}
        </text>
      ))}
      <text x={CX1} y={CY1 + 30} textAnchor="end" className={S.text}>
        similarity to nearest training point
      </text>
      <text x={CX0} y={CY0 - 10} className={S.text}>
        typical error (illustrative)
      </text>
      <path d={CURVE} className={S.fit} strokeWidth={1.75} strokeLinejoin="round" />
      <line
        x1={ex.start[0]}
        y1={ex.start[1]}
        x2={ex.end[0]}
        y2={ex.end[1]}
        opacity={seg(t, 0.86, 0.95)}
        className={S.line}
        strokeDasharray="2 4"
      />
      <g opacity={moving}>
        <line
          x1={pt.x}
          y1={pt.y}
          x2={near[0]}
          y2={near[1]}
          className={S.heldLine}
          strokeWidth={1.25}
        />
        <path
          d={`M${pt.x} ${pt.y - bar} V${pt.y + bar} M${pt.x - 4} ${pt.y - bar} H${pt.x + 4} M${pt.x - 4} ${pt.y + bar} H${pt.x + 4}`}
          className={S.heldLine}
          strokeWidth={1.5}
        />
        <Marker cx={pt.x} cy={pt.y} r={5.5} out={out} width={1.75} />
        <Marker
          cx={CX(pt.similarity)}
          cy={CY(typicalError(pt.similarity))}
          r={4.5}
          out={out}
          width={1.5}
        />
      </g>
      {ex.stamps.map((st, k) => {
        const stampOut = st.similarity < threshold;
        return (
          <g key={st.s} opacity={t >= 0.18 && s >= st.s - 1e-9 ? 1 : 0}>
            <Marker cx={st.x} cy={st.y} r={4} out={stampOut} width={1.5} />
            <text x={st.x + 9} y={st.y - 7} className={S.textStrong}>
              {k + 1}
            </text>
            <Marker
              cx={CX(st.similarity)}
              cy={CY(typicalError(st.similarity))}
              r={3.5}
              out={stampOut}
              width={1.5}
            />
            <text
              x={CX(st.similarity) + 8}
              y={CY(typicalError(st.similarity)) - 7}
              className={S.textStrong}
            >
              {k + 1} · {st.similarity.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
