import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const FOREST_MS = 5600;
export const FOREST_CAPTION =
  "Each tree is trained on a different random sample of the data and asks its own sequence of questions. The forest's prediction is the average of the trees. For regression, Studio reports the spread across trees as the uncertainty. Showing 4 of 500 trees.";

type Pt = [number, number];

const CENTERS = [54, 138, 222, 306];
/** Which branch each tree takes at its two levels: 0 left, 1 right. */
const PATHS: Pt[] = [
  [0, 1],
  [1, 0],
  [0, 0],
  [1, 1],
];
const LEAF_VALUES = [
  [0.52, 0.58, 0.7, 0.77],
  [0.61, 0.66, 0.74, 0.83],
  [0.63, 0.69, 0.55, 0.79],
  [0.57, 0.66, 0.72, 0.81],
];
const LINE_Y = 136;
const X = (v: number) => 80 + ((v - 0.4) / 0.6) * 250;

const TREES = CENTERS.map((cx, i) => {
  const root: Pt = [cx, 30];
  const mids: Pt[] = [
    [cx - 20, 56],
    [cx + 20, 56],
  ];
  const leaves: Pt[] = [
    [cx - 30, 82],
    [cx - 10, 82],
    [cx + 10, 82],
    [cx + 30, 82],
  ];
  const [a, b] = PATHS[i];
  return { root, mids, leaves, a, leafIndex: a * 2 + b, value: LEAF_VALUES[i][a * 2 + b] };
});
const VALUES = TREES.map((tree) => tree.value);
const MEAN = VALUES.reduce((s, v) => s + v, 0) / VALUES.length;
const SD = Math.sqrt(VALUES.reduce((s, v) => s + (v - MEAN) ** 2, 0) / VALUES.length);
const LO = Math.min(...VALUES);
const HI = Math.max(...VALUES);

function Partial({ from, to, p }: { from: Pt; to: Pt; p: number }) {
  return (
    <line
      x1={from[0]}
      y1={from[1]}
      x2={lerp(from[0], to[0], p)}
      y2={lerp(from[1], to[1], p)}
      opacity={p > 0 ? 1 : 0}
      className={S.measuredLine}
      strokeWidth={2.25}
      strokeLinecap="round"
    />
  );
}

export function ForestFigure({ t }: { t: number }) {
  return (
    <svg
      viewBox="0 0 360 172"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="Four decision trees each route one input to a leaf. Their four predictions, 0.58, 0.74, 0.63 and 0.81, average to 0.69, with a spread of plus or minus 0.09."
    >
      <text x={356} y={11} textAnchor="end" className={S.small}>
        4 of 500 trees
      </text>
      {TREES.map((tree, i) => {
        const s = 0.14 + i * 0.07;
        const p1 = ease(seg(t, s, s + 0.1));
        const p2 = ease(seg(t, s + 0.1, s + 0.2));
        const mid = tree.mids[tree.a];
        const leaf = tree.leaves[tree.leafIndex];
        const d = ease(seg(t, 0.58 + i * 0.04, 0.72 + i * 0.04));
        return (
          <g key={tree.root[0]}>
            <g opacity={ease(seg(t, i * 0.02, 0.1 + i * 0.02))}>
              {tree.mids.map((n) => (
                <line
                  key={`r${n[0]}`}
                  x1={tree.root[0]}
                  y1={tree.root[1]}
                  x2={n[0]}
                  y2={n[1]}
                  className={S.line}
                  strokeWidth={1.5}
                />
              ))}
              {tree.leaves.map((n, j) => (
                <line
                  key={`l${n[0]}`}
                  x1={tree.mids[j >> 1][0]}
                  y1={tree.mids[j >> 1][1]}
                  x2={n[0]}
                  y2={n[1]}
                  className={S.line}
                  strokeWidth={1.5}
                />
              ))}
              <Partial from={tree.root} to={mid} p={p1} />
              <Partial from={mid} to={leaf} p={p2} />
              <circle
                cx={tree.root[0]}
                cy={tree.root[1]}
                r={5}
                className={t >= s ? S.nodeOn : S.node}
                strokeWidth={t >= s ? 2 : 1.5}
              />
              {tree.mids.map((n, j) => (
                <circle
                  key={`m${n[0]}`}
                  cx={n[0]}
                  cy={n[1]}
                  r={4.5}
                  className={j === tree.a && p1 >= 1 ? S.measured : S.node}
                  strokeWidth={1.5}
                />
              ))}
              {tree.leaves.map((n, j) => (
                <circle
                  key={`f${n[0]}`}
                  cx={n[0]}
                  cy={n[1]}
                  r={3.5}
                  className={j === tree.leafIndex && p2 >= 1 ? S.measured : S.node}
                  strokeWidth={1.5}
                />
              ))}
            </g>
            <text
              x={leaf[0]}
              y={97}
              textAnchor="middle"
              className={S.smallStrong}
              opacity={seg(t, s + 0.2, s + 0.26)}
            >
              {tree.value.toFixed(2)}
            </text>
            <circle
              cx={lerp(leaf[0], X(tree.value), d)}
              cy={lerp(leaf[1] + 20, LINE_Y, d)}
              r={4}
              opacity={d > 0 ? 1 : 0}
              className={S.pred}
              strokeWidth={1.5}
            />
          </g>
        );
      })}
      <line x1={X(0.4)} y1={LINE_Y} x2={X(1)} y2={LINE_Y} className={S.line} />
      {[0.4, 0.6, 0.8, 1].map((v) => (
        <g key={v}>
          <line x1={X(v)} y1={LINE_Y - 3} x2={X(v)} y2={LINE_Y + 3} className={S.line} />
          <text x={X(v)} y={LINE_Y + 14} textAnchor="middle" className={S.text}>
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      <text x={72} y={LINE_Y + 3.5} textAnchor="end" className={S.text}>
        prediction
      </text>
      <g opacity={seg(t, 0.84, 0.92)}>
        <line x1={X(LO)} y1={160} x2={X(HI)} y2={160} className={S.ink} strokeWidth={1.25} />
        <line x1={X(LO)} y1={156} x2={X(LO)} y2={164} className={S.ink} strokeWidth={1.25} />
        <line x1={X(HI)} y1={156} x2={X(HI)} y2={164} className={S.ink} strokeWidth={1.25} />
        <text x={X(LO) - 6} y={163.5} textAnchor="end" className={S.text}>
          spread ±{SD.toFixed(2)}
        </text>
      </g>
      <g opacity={seg(t, 0.88, 1)}>
        <line
          x1={X(MEAN)}
          y1={118}
          x2={X(MEAN)}
          y2={146}
          className={S.ink}
          strokeWidth={2.25}
          strokeLinecap="round"
        />
        <text x={X(MEAN)} y={112} textAnchor="middle" className={S.textStrong}>
          average {MEAN.toFixed(2)}
        </text>
      </g>
    </svg>
  );
}
