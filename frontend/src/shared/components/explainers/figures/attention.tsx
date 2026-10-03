import { clamp, ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const ATTENTION_MS = 7200;
export const ATTENTION_CAPTION =
  "The input is split into tokens. Each token weighs every other token when building its representation; arc width shows the weight. This step is called attention. The encoder arrives pretrained, and training on your data either fine-tunes it or, with the encoder frozen, trains only the output layer.";

const WORDS = ["every", "token", "looks", "at", "every", "other", "token"];
const CHAR_W = 6.1;
const PAD = 6;
const GAP = 7;
const WIDTHS = WORDS.map((w) => w.length * CHAR_W + PAD * 2);
const START_X = (360 - WIDTHS.reduce((s, w) => s + w, 0) - GAP * (WORDS.length - 1)) / 2;
const TOKENS = WORDS.map((word, i) => {
  const x = START_X + WIDTHS.slice(0, i).reduce((s, w) => s + w + GAP, 0);
  return { word, x, w: WIDTHS[i], c: x + WIDTHS[i] / 2 };
});
/** Attention weights from three focus tokens to every token (self excluded). */
const WEIGHTS: Record<number, number[]> = {
  1: [0.3, 0, 0.15, 0.05, 0.1, 0.1, 0.3],
  5: [0.05, 0.35, 0.1, 0.05, 0.15, 0, 0.3],
  2: [0.1, 0.45, 0, 0.25, 0.05, 0.05, 0.1],
};
/** [focus token, window start, window end]; the last focus stays on. */
const FOCI: [number, number, number][] = [
  [1, 0.1, 0.34],
  [5, 0.36, 0.6],
  [2, 0.62, 0.86],
];

export function AttentionFigure({ t }: { t: number }) {
  let focus: number | null = null;
  let env = 0;
  for (const [k, [f, a, b]] of FOCI.entries()) {
    const lastFocus = k === FOCI.length - 1;
    const e = lastFocus
      ? seg(t, a, a + 0.06)
      : Math.min(seg(t, a, a + 0.06), 1 - seg(t, b - 0.04, b));
    if (t >= a && (lastFocus || t < b)) {
      focus = f;
      env = e;
    }
  }
  const flow = seg(t, 0.88, 0.97);

  return (
    <svg
      viewBox="0 0 360 172"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="Seven tokens in a sentence. Arcs show how strongly one token attends to each of the others. The pretrained encoder feeds an output layer trained on your data."
    >
      <rect x={6} y={4} width={348} height={104} rx={8} className={S.line} strokeWidth={1.25} />
      <text x={14} y={101} className={S.small}>
        encoder, pretrained on ~1.1 billion molecules
      </text>
      {TOKENS.map((token, j) => {
        if (focus === null || j === focus) return null;
        const w = WEIGHTS[focus][j];
        const a = TOKENS[focus].c;
        const b = token.c;
        const h = 12 + Math.abs(a - b) * 0.3;
        return (
          <path
            key={token.x}
            d={`M${a} 68 Q${(a + b) / 2} ${68 - h} ${b} 68`}
            strokeWidth={0.6 + w * 7}
            strokeLinecap="round"
            opacity={env * clamp(0.25 + w * 1.6)}
            className={S.measuredLine}
          />
        );
      })}
      {TOKENS.map((token, i) => (
        <g key={token.x}>
          <rect
            x={token.x}
            y={70}
            width={token.w}
            height={18}
            rx={4}
            className={focus === i && env > 0.05 ? S.boxOn : S.box}
            strokeWidth={focus === i && env > 0.05 ? 1.5 : 1.25}
          />
          <text x={token.c} y={82.5} textAnchor="middle" className={S.textStrong}>
            {token.word}
          </text>
        </g>
      ))}
      <line x1={180} y1={108} x2={180} y2={122} className={S.line} />
      <rect
        x={72}
        y={122}
        width={216}
        height={22}
        rx={5}
        className={t >= 0.95 ? S.boxOn : S.box}
        strokeWidth={t >= 0.95 ? 1.5 : 1.25}
      />
      <text x={180} y={136.5} textAnchor="middle" className={S.textStrong}>
        output layer, trained on your data
      </text>
      <line x1={180} y1={144} x2={180} y2={154} className={S.line} />
      <text x={180} y={166} textAnchor="middle" className={t >= 0.97 ? S.textStrong : S.text}>
        prediction
      </text>
      <circle
        cx={180}
        cy={lerp(108, 122, ease(flow))}
        r={3}
        opacity={flow > 0 && flow < 1 ? 1 : 0}
        className={S.measured}
      />
    </svg>
  );
}
