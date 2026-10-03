import { gpExample } from "../math/gaussian-process";
import { ease, easeInOut, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const GP_MS = 7000;
export const GP_CAPTION =
  "Before any data, every value is equally plausible. Each observation pins the curve near itself, and the shaded band, two standard deviations wide, narrows there. Far from the data the band stays wide. Shown in one dimension; Studio measures closeness with Tanimoto similarity.";

const EX = gpExample();
const PX = (x: number) => 40 + x * 300;
const PY = (y: number) => 144 - y * 120;
const GAP_X = 0.57;
const NEAR_X = 0.31;
const last = EX.grid.length - 1;
/** When observation k (1-based) appears. */
const appear = (k: number) => 0.1 + (k - 1) * 0.15;

export function GaussianProcessFigure({ t }: { t: number }) {
  let n = 0;
  let p = 0;
  for (let q = 1; q <= EX.xs.length; q++) {
    const m = easeInOut(seg(t, appear(q) + 0.03, appear(q) + 0.13));
    if (m >= 1) n = q;
    else {
      p = m;
      break;
    }
  }
  const from = EX.states[n];
  const to = EX.states[Math.min(n + 1, EX.xs.length)];
  const mean = from.mean.map((m, i) => lerp(m, to.mean[i], p));
  const sd = from.sd.map((s, i) => lerp(s, to.sd[i], p));
  const upper = EX.grid
    .map((g, i) => `${i ? "L" : "M"}${PX(g)} ${PY(mean[i] + 2 * sd[i])}`)
    .join(" ");
  const lower = EX.grid
    .map((g, i) => `L${PX(g)} ${PY(mean[i] - 2 * sd[i])}`)
    .reverse()
    .join(" ");
  const line = EX.grid.map((g, i) => `${i ? "L" : "M"}${PX(g)} ${PY(mean[i])}`).join(" ");
  const top = (x: number) => {
    const i = Math.round(x * last);
    return PY(mean[i] + 2 * sd[i]);
  };
  const shown = seg(t, 0, 0.08);

  return (
    <svg
      viewBox="0 0 360 172"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="A Gaussian process fit to five points. The shaded uncertainty band is narrow near observed points and wide in the gap between them."
    >
      <path d={`${upper} ${lower} Z`} opacity={shown} className={S.band} />
      <path d={line} opacity={shown} className={S.fit} strokeWidth={1.75} strokeLinejoin="round" />
      {EX.xs.map((x, i) => {
        const a = ease(seg(t, appear(i + 1), appear(i + 1) + 0.04));
        return (
          <circle
            key={x}
            cx={PX(x)}
            cy={PY(EX.ys[i])}
            r={3.4 * (0.4 + 0.6 * a)}
            opacity={a}
            className={S.measured}
          />
        );
      })}
      <g opacity={seg(t, 0.86, 0.98)}>
        <text x={PX(GAP_X)} y={12} textAnchor="middle" className={S.textStrong}>
          wide: no data nearby
        </text>
        <text x={40} y={12} className={S.textStrong}>
          narrow: near data
        </text>
        <line
          x1={PX(GAP_X)}
          x2={PX(GAP_X)}
          y1={16}
          y2={top(GAP_X) - 3}
          className={S.ink}
          strokeWidth={1}
        />
        <line
          x1={PX(NEAR_X)}
          x2={PX(NEAR_X)}
          y1={16}
          y2={top(NEAR_X) - 4}
          className={S.ink}
          strokeWidth={1}
        />
      </g>
    </svg>
  );
}
