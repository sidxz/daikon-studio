import { gauss, mulberry32 } from "./prng";
import { lerp } from "./tween";

export interface PathPoint {
  s: number;
  x: number;
  y: number;
  similarity: number;
}
export interface DomainExample {
  training: [number, number][];
  /** Distance at which similarity falls to the threshold: the radius of the domain region. */
  reach: number;
  start: [number, number];
  end: [number, number];
  path: PathPoint[];
  stamps: PathPoint[];
}

const SCALE = 45;
/** Illustrative typical error at a given similarity (0..1). Not fitted to any protocol. */
export const typicalError = (s: number) => 0.12 + 0.86 * (1 - s) ** 2.2;

export function nearestPoint(points: [number, number][], x: number, y: number) {
  let distance = Number.POSITIVE_INFINITY;
  let index = 0;
  points.forEach(([a, b], j) => {
    const d = Math.hypot(x - a, y - b);
    if (d < distance) {
      distance = d;
      index = j;
    }
  });
  return { distance, index };
}

/** A query walks away from three training clusters; stamps at start, near the edge, and outside. */
export function domainExample(threshold: number): DomainExample {
  const rand = mulberry32(9);
  const training: [number, number][] = [];
  for (const [cx, cy, n, s] of [
    [86, 92, 24, 16],
    [146, 172, 26, 18],
    [206, 96, 14, 13],
  ]) {
    for (let i = 0; i < n; i++) training.push([cx + gauss(rand) * s, cy + gauss(rand) * s * 0.85]);
  }
  const start: [number, number] = [118, 132];
  const end: [number, number] = [318, 214];
  const path = Array.from({ length: 241 }, (_, i) => {
    const s = i / 240;
    const x = lerp(start[0], end[0], s);
    const y = lerp(start[1], end[1], s);
    return { s, x, y, similarity: Math.exp(-nearestPoint(training, x, y).distance / SCALE) };
  });
  const edge = path.findIndex((p) => p.similarity < threshold + 0.06);
  const stamps = [0, Math.max(1, edge), path.length - 1].map((i) => path[i]);
  return { training, reach: -SCALE * Math.log(threshold), start, end, path, stamps };
}
