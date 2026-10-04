"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * The clock behind every explainer figure. `t` runs 0 -> 1 once, the first
 * time the figure is a third in view, and again on replay(). It rests at 1, so the first render, a server render and
 * reduced motion all show the complete final frame.
 */
export function useTimeline<T extends Element>(durationMs: number) {
  const ref = useRef<T>(null);
  const [t, setT] = useState(1);
  const frame = useRef(0);

  const replay = useCallback(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    cancelAnimationFrame(frame.current);
    let start: number | null = null;
    const step = (now: number) => {
      start ??= now;
      const next = Math.min(1, (now - start) / durationMs);
      setT(next);
      frame.current = next < 1 ? requestAnimationFrame(step) : 0;
    };
    setT(0);
    frame.current = requestAnimationFrame(step);
  }, [durationMs]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let observer: IntersectionObserver | undefined;
    if (typeof IntersectionObserver !== "undefined") {
      observer = new IntersectionObserver(
        (entries) => {
          if (!entries.some((e) => e.isIntersecting)) return;
          observer?.disconnect();
          replay();
        },
        { threshold: 0.35 },
      );
      observer.observe(el);
    }
    return () => {
      observer?.disconnect();
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [replay]);

  return { ref, t, replay };
}
