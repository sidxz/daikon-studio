/** Timeline helpers. Every explainer figure is a pure function of t in [0, 1]. */
export const clamp = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
/** Progress of t through the window [a, b], clamped to [0, 1]. */
export const seg = (t: number, a: number, b: number) => clamp((t - a) / (b - a));
export const ease = (x: number) => 1 - (1 - x) ** 3;
export const easeInOut = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
