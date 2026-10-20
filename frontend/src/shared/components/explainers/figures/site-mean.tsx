import { ease, lerp, seg } from "../math/tween";
import { S } from "./styles";

export const SITE_MEAN_MS = 6400;
export const SITE_MEAN_CAPTION =
  "Every measured variant is sorted by the residue it changes, not by what replaced it. Each position keeps one number, the average of its variants, and that average is the prediction for any new variant at that position. A model worth using has to beat this.";

/** Three mutated positions, each with the fitness of the variants measured there.
 *  Spread within a column is the point: the baseline throws it away. */
const SITES: { label: string; values: number[] }[] = [
  { label: "24", values: [0.82, 0.61, 0.55, 0.3] },
  { label: "39", values: [0.46, 0.2, 0.12] },
  { label: "71", values: [0.9, 0.74, 0.68, 0.52, 0.41] },
];

const COL_X = [72, 180, 288];
const TOP = 26;
const BOTTOM = 104;
/** Fitness 0..1 to a y inside the plot, higher fitness higher up. */
const y = (v: number) => lerp(BOTTOM, TOP, v);
const MEANS = SITES.map((s) => s.values.reduce((a, b) => a + b, 0) / s.values.length);

export function SiteMeanFigure({ t }: { t: number }) {
  // Dots appear, then slide onto their column's mean, which is then read off as
  // the prediction for an unmeasured variant at that site.
  const appear = seg(t, 0.04, 0.3);
  const collapse = ease(seg(t, 0.36, 0.66));
  const predict = seg(t, 0.74, 0.96);

  return (
    <svg
      viewBox="0 0 360 150"
      className="block h-auto w-full overflow-visible"
      role="img"
      aria-label="Variants grouped by the residue position they mutate. The measured values at each position collapse onto their average, and that average becomes the prediction for a new variant at the same position."
    >
      <line x1={40} y1={BOTTOM + 8} x2={330} y2={BOTTOM + 8} className={S.line} />
      <text x={40} y={TOP - 12} className={S.small}>
        fitness
      </text>

      {SITES.map((site, i) => {
        const mean = MEANS[i];
        const shown = Math.round(appear * site.values.length);
        return (
          <g key={site.label}>
            <text x={COL_X[i]} y={BOTTOM + 22} textAnchor="middle" className={S.small}>
              position {site.label}
            </text>
            {site.values.map((v, j) => {
              if (j >= shown) return null;
              // Each dot travels from where it was measured to its column's average.
              const cy = lerp(y(v), y(mean), collapse);
              const spread = lerp(14, 0, collapse);
              const cx = COL_X[i] + (j - (site.values.length - 1) / 2) * (spread / 2);
              return (
                <g key={`${site.label}-${j}`}>
                  {/* Where the variant was actually measured, left behind as it
                      collapses. Without it the figure comes to rest showing one dot a
                      position, and the spread this baseline discards -- the whole
                      reason it is only a floor -- would be visible mid-animation only. */}
                  <circle
                    cx={COL_X[i] + (j - (site.values.length - 1) / 2) * 7}
                    cy={y(v)}
                    r={2.5}
                    className={S.measured}
                    opacity={0.22 * collapse}
                  />
                  <circle cx={cx} cy={cy} r={3} className={S.measured} />
                </g>
              );
            })}
            {/* The kept number: one average per position. */}
            <line
              x1={COL_X[i] - 20}
              y1={y(mean)}
              x2={COL_X[i] + 20}
              y2={y(mean)}
              className={S.fit}
              strokeWidth={1.5}
              opacity={collapse}
            />
            {predict > 0 && (
              <>
                <circle
                  cx={COL_X[i] + 30}
                  cy={y(mean)}
                  r={3.5}
                  className={S.pred}
                  strokeWidth={1.5}
                  opacity={predict}
                />
                <text
                  x={COL_X[i] + 30}
                  y={y(mean) - 9}
                  textAnchor="middle"
                  className={S.smallStrong}
                  opacity={predict}
                >
                  new
                </text>
              </>
            )}
          </g>
        );
      })}

      <text x={40} y={142} className={S.small} opacity={predict}>
        a new variant at a position is predicted its position's average
      </text>
    </svg>
  );
}
