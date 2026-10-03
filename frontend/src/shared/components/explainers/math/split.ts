import { gauss, mulberry32, shuffle } from "./prng";

export interface SplitPoint {
  x: number;
  y: number;
  group: number;
}
export interface SplitLink {
  test: number;
  train: number;
  distance: number;
}
export interface SplitMode {
  test: number[];
  links: SplitLink[];
  medianSimilarity: number;
}
export interface SplitExample {
  points: SplitPoint[];
  random: SplitMode;
  scaffold: SplitMode;
}

/** On-screen distance to a 0..1 stand-in for Tanimoto similarity. Illustrative only. */
export const similarityAt = (distance: number) => Math.exp(-distance / 45);

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
}

function mode(points: SplitPoint[], test: number[]): SplitMode {
  const held = new Set(test);
  const links = test.map((i) => {
    let train = -1;
    let distance = Number.POSITIVE_INFINITY;
    points.forEach((p, j) => {
      if (held.has(j)) return;
      const d = Math.hypot(points[i].x - p.x, points[i].y - p.y);
      if (d < distance) {
        distance = d;
        train = j;
      }
    });
    return { test: i, train, distance };
  });
  return { test, links, medianSimilarity: median(links.map((l) => similarityAt(l.distance))) };
}

/** 110 points in five groups; one fifth held out at random, or one whole group (scaffold). */
export function splitExample(): SplitExample {
  const rand = mulberry32(11);
  const groups: [number, number, number][] = [
    [78, 70, 24],
    [196, 52, 20],
    [318, 112, 22],
    [112, 196, 18],
    [262, 206, 26],
  ];
  const points: SplitPoint[] = [];
  groups.forEach(([cx, cy, n], group) => {
    for (let i = 0; i < n; i++)
      points.push({ x: cx + gauss(rand) * 17, y: cy + gauss(rand) * 14, group });
  });
  const random = shuffle([...points.keys()], mulberry32(5)).slice(0, 22);
  const scaffold = [...points.keys()].filter((i) => points[i].group === 2);
  return { points, random: mode(points, random), scaffold: mode(points, scaffold) };
}
