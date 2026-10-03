import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const MESSAGE_PASSING_MS = 7600;
export const MESSAGE_PASSING_CAPTION =
  "In each round, every node collects messages from its neighbors. After three rounds, the outlined node carries information from nodes up to three steps away; the node four steps away is still unseen. The node states are then summed into one vector that the predictor reads. Three rounds is the default depth.";

type Pt = [number, number];

const NODES: Pt[] = [
  [34, 92],
  [78, 52],
  [82, 130],
  [128, 90],
  [172, 42],
  [178, 130],
  [222, 86],
];
const EDGES: Pt[] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 3],
  [3, 4],
  [3, 5],
  [4, 6],
  [5, 6],
];
/** Steps from the outlined node 0. */
const HOP = [0, 1, 1, 2, 3, 3, 4];
/** Both directions of every edge carry a message each round. */
const MESSAGES = EDGES.flatMap(([a, b]) => [
  [a, b],
  [b, a],
]);
const SUM: Pt = [272, 86];
const SHADES = [0.9, 0.35, 0.65, 0.2, 0.8];
const roundStart = (r: number) => 0.1 + (r - 1) * 0.22;
const roundEnd = (r: number) => roundStart(r) + 0.18;

export function MessagePassingFigure({ t }: { t: number }) {
  let round = 0;
  for (let r = 1; r <= 3; r++) if (t >= roundStart(r)) round = r;
  const label =
    t < roundStart(1)
      ? "start: each node knows only itself"
      : t >= 0.76
        ? "readout"
        : `round ${round} of 3`;
  const appear = seg(t, 0, 0.08);
  // Progress of the messages in flight, or -1 between rounds.
  let flight = -1;
  for (let r = 1; r <= 3; r++) {
    const q = seg(t, roundStart(r), roundStart(r) + 0.13);
    if (q > 0 && q < 1) flight = ease(q);
  }

  return (
    <svg
      viewBox="0 0 360 172"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="A seven-node graph. Over three rounds, messages pass along edges and the highlighted node's view grows to nodes three steps away. All node states are then summed into one learned vector."
    >
      <g opacity={seg(t, 0.76, 0.86)}>
        <path d="M242 34 H250 V138 H242" className={S.line} strokeWidth={1.5} />
        <line x1={250} y1={SUM[1]} x2={SUM[0] - 9} y2={SUM[1]} className={S.line} />
        <circle cx={SUM[0]} cy={SUM[1]} r={9} className={S.box} strokeWidth={1.25} />
        <path
          d={`M${SUM[0] - 4} ${SUM[1]} H${SUM[0] + 4} M${SUM[0]} ${SUM[1] - 4} V${SUM[1] + 4}`}
          className={S.ink}
          strokeWidth={1.5}
        />
        <line x1={SUM[0] + 10} y1={SUM[1]} x2={289} y2={SUM[1]} className={S.line} />
        {SHADES.map((shade, i) => (
          <g key={shade}>
            <rect
              x={292 + i * 12.5}
              y={79}
              width={11}
              height={14}
              rx={2}
              className={S.box}
              strokeWidth={1.25}
            />
            <rect
              x={292 + i * 12.5}
              y={79}
              width={11}
              height={14}
              rx={2}
              fillOpacity={shade}
              opacity={seg(t, 0.84 + i * 0.025, 0.9 + i * 0.025)}
              className={S.measured}
            />
          </g>
        ))}
        <text x={322} y={70} textAnchor="middle" className={S.smallStrong}>
          learned vector
        </text>
      </g>
      {EDGES.map(([a, b]) => (
        <line
          key={`${a}-${b}`}
          x1={NODES[a][0]}
          y1={NODES[a][1]}
          x2={NODES[b][0]}
          y2={NODES[b][1]}
          className={S.line}
          strokeWidth={1.5}
        />
      ))}
      {MESSAGES.map(([u, v]) => (
        <circle
          key={`${u}>${v}`}
          r={2.4}
          cx={lerp(NODES[u][0], NODES[v][0], Math.max(0, flight))}
          cy={lerp(NODES[u][1], NODES[v][1], Math.max(0, flight))}
          opacity={flight < 0 ? 0 : Math.sin(Math.PI * flight) * 0.95}
          className={S.pulse}
        />
      ))}
      {NODES.map(([x, y], i) => {
        const h = HOP[i];
        const reached = h === 0 ? appear : h <= 3 ? seg(t, roundEnd(h) - 0.04, roundEnd(h)) : 0;
        return (
          <g key={`n${x}`}>
            <circle cx={x} cy={y} r={8} opacity={appear} className={S.node} strokeWidth={1.5} />
            <circle cx={x} cy={y} r={8} opacity={reached * (1 - h * 0.18)} className={S.measured} />
          </g>
        );
      })}
      <circle
        cx={NODES[0][0]}
        cy={NODES[0][1]}
        r={12}
        className={S.measuredLine}
        strokeWidth={1.5}
      />
      <text x={NODES[6][0]} y={NODES[6][1] + 24} textAnchor="middle" className={S.small}>
        unseen
      </text>
      <text x={8} y={12} className={S.textStrong}>
        {label}
      </text>
    </svg>
  );
}
