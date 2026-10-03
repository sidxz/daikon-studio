import { splitExample } from "../math/split";
import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export type SplitKind = "scaffold" | "random";

export const SPLIT_MS = 5400;

export function splitCaption(strategy: SplitKind): string {
  return strategy === "scaffold"
    ? "Scaffold split: whole groups are held out, so each test point is far from anything the model trained on (long amber links). This approximates predicting a new chemical series. Each cluster stands for one Bemis–Murcko scaffold; similarity here is computed from on-screen distance."
    : "Random split: test points are drawn from every group, so most sit next to a near-identical training point (short amber links). The test then rewards memorization, and scores look better than they will be on new chemistry. Each cluster stands for one Bemis–Murcko scaffold; similarity here is computed from on-screen distance.";
}

const EX = splitExample();
const N = EX.points.length;
const RX = 440;
const RW = 180;
const LEGEND_Y = 214;
const BARS: [SplitKind, string, number][] = [
  ["random", "Random split", 92],
  ["scaffold", "Scaffold split", 146],
];

export function SplitFigure({ t, strategy }: { t: number; strategy: SplitKind }) {
  const mode = EX[strategy];
  const order = new Map(mode.test.map((i, k) => [i, k]));
  const grow = ease(seg(t, 0.8, 0.96));

  return (
    <svg
      viewBox="0 0 640 270"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="110 points in five clusters. One fifth are held out as the test set, and each test point is joined to its nearest training point. A bar chart compares the median similarity under each split."
    >
      {mode.links.map((link, k) => {
        const q = ease(seg(t, 0.46 + (k / 22) * 0.3, 0.54 + (k / 22) * 0.3));
        const a = EX.points[link.test];
        const b = EX.points[link.train];
        return (
          <line
            key={link.test}
            x1={a.x}
            y1={a.y}
            x2={lerp(a.x, b.x, q)}
            y2={lerp(a.y, b.y, q)}
            opacity={q > 0 ? 0.9 : 0}
            className={S.heldLine}
            strokeWidth={1.25}
          />
        );
      })}
      {EX.points.map((p, i) => {
        const appear = ease(seg(t, (i / N) * 0.12, (i / N) * 0.12 + 0.05));
        const k = order.get(i);
        const c = k === undefined ? 0 : seg(t, 0.16 + (k / 22) * 0.24, 0.2 + (k / 22) * 0.24);
        return (
          <circle
            key={`${p.x}:${p.y}`}
            cx={p.x}
            cy={p.y}
            r={3 + 1.6 * Math.sin(Math.PI * c) + (c >= 1 ? 0.4 : 0)}
            opacity={appear}
            className={c > 0.5 ? S.held : S.measured}
          />
        );
      })}
      <text x={RX} y={28} className={S.textStrong}>
        Similarity of each test point
      </text>
      <text x={RX} y={42} className={S.textStrong}>
        to its nearest training point
      </text>
      <text x={RX} y={56} className={S.small}>
        median, 1 = identical
      </text>
      {BARS.map(([kind, label, y]) => {
        const active = kind === strategy;
        const value = EX[kind].medianSimilarity * (active ? grow : 1);
        return (
          <g key={kind}>
            <text x={RX} y={y - 8} className={S.text}>
              {label}
            </text>
            <text x={RX + RW} y={y - 8} textAnchor="end" className={S.textStrong}>
              {value.toFixed(2)}
            </text>
            <rect x={RX} y={y} width={RW} height={14} rx={3} className={S.box} strokeWidth={1} />
            <rect
              x={RX}
              y={y}
              width={Math.max(0, value * RW)}
              height={14}
              rx={3}
              className={active ? S.held : "fill-muted-foreground/30"}
            />
          </g>
        );
      })}
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={RX + v * RW} y={178} textAnchor="middle" className={S.small}>
          {v.toFixed(1)}
        </text>
      ))}
      <circle cx={RX + 4} cy={LEGEND_Y} r={3.5} className={S.measured} />
      <text x={RX + 14} y={LEGEND_Y + 3.5} className={S.text}>
        training set (80%)
      </text>
      <circle cx={RX + 4} cy={LEGEND_Y + 18} r={3.5} className={S.held} />
      <text x={RX + 14} y={LEGEND_Y + 21.5} className={S.text}>
        test set (20%)
      </text>
      <line
        x1={RX}
        y1={LEGEND_Y + 36}
        x2={RX + 9}
        y2={LEGEND_Y + 36}
        className={S.heldLine}
        strokeWidth={1.5}
      />
      <text x={RX + 14} y={LEGEND_Y + 39.5} className={S.text}>
        link to nearest training point
      </text>
    </svg>
  );
}
