import { boostingExample } from "../math/boosting";
import { easeInOut, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const BOOSTING_MS = 7800;
export const BOOSTING_CAPTION =
  "The model starts by predicting the average. Each round adds one small tree, fitted to the errors that remain (amber). The errors shrink round by round. Showing 6 of 400 rounds, in one dimension.";

const EX = boostingExample();
const ROUNDS = EX.stages.length - 1;
const START = 0.12;
const WIDTH = 0.14;
const PX = (x: number) => 40 + x * 300;
const PY = (y: number) => 146 - y * 112;

export function BoostingFigure({ t }: { t: number }) {
  // Completed rounds k, and progress p through the round being drawn.
  let k = 0;
  let p = 0;
  for (let m = 1; m <= ROUNDS; m++) {
    const q = easeInOut(seg(t, START + (m - 1) * WIDTH + 0.05, START + (m - 1) * WIDTH + 0.13));
    if (q >= 1) k = m;
    else {
      p = q;
      break;
    }
  }
  const at = (stages: number[][], j: number) =>
    k < ROUNDS ? lerp(stages[k][j], stages[k + 1][j], p) : stages[k][j];
  const steps = EX.stages[0].map((_, j) => at(EX.stages, j));
  let path = `M${PX(0)} ${PY(steps[0])}`;
  steps.forEach((v, j) => {
    path += ` V${PY(v)} H${PX(EX.bounds[j + 1])}`;
  });
  const fit = EX.xs.map((_, i) => at(EX.pointStages, i));
  const inRound = k < ROUNDS ? seg(t, START + k * WIDTH, START + k * WIDTH + 0.06) : 0;
  const glow = Math.sin(Math.PI * inRound);
  const meanError = EX.ys.reduce((s, y, i) => s + Math.abs(y - fit[i]), 0) / EX.ys.length;
  const shown = Math.min(ROUNDS, k + (p > 0 || inRound > 0 ? 1 : 0));

  return (
    <svg
      viewBox="0 0 360 172"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="Gradient boosting on 14 points. The fit starts as a flat line at the average; each round adds a small step that reduces the remaining errors, shown as amber lines."
    >
      <line x1={40} y1={150} x2={340} y2={150} className={S.line} />
      {EX.xs.map((x, i) => (
        <line
          key={`r${x}`}
          x1={PX(x)}
          x2={PX(x)}
          y1={PY(EX.ys[i])}
          y2={PY(fit[i])}
          opacity={seg(t, 0.08, 0.12) * (0.55 + 0.45 * glow)}
          className={S.heldLine}
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      ))}
      <path
        d={path}
        opacity={seg(t, 0.06, 0.12)}
        className={S.fit}
        strokeWidth={1.75}
        strokeLinejoin="round"
      />
      {EX.xs.map((x, i) => {
        const a = (i / EX.xs.length) * 0.06;
        return (
          <circle
            key={`p${x}`}
            cx={PX(x)}
            cy={PY(EX.ys[i])}
            r={3.2}
            opacity={seg(t, a, a + 0.03)}
            className={S.measured}
          />
        );
      })}
      <text x={40} y={12} className={S.textStrong}>
        {t < START ? "start: predict the average" : `round ${Math.max(shown, 1)} of ${ROUNDS}`}
      </text>
      <text x={340} y={12} textAnchor="end" className={S.text}>
        mean error {meanError.toFixed(3)}
      </text>
    </svg>
  );
}
