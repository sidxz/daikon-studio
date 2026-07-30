"use client";

import { useId } from "react";

/**
 * Brand mark -- a hexagon drawn half solid, half dashed.
 *
 * The solid half is what was measured; the dashed half is what was predicted.
 * That distinction is the entire product thesis, so it belongs in the mark. The
 * hexagon keeps the family resemblance to ChemCellar's own hex-lens, and the
 * gradient is the shared suite spectrum.
 *
 * `useId` for the gradient id so several instances on one page (sidebar and
 * login, say) do not collide on a duplicate `id`.
 */
export function LogoMark({ className }: { className?: string }) {
  const id = useId();
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <title>DAIKON Studio</title>
      <defs>
        <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="8" y1="25" x2="24" y2="7">
          <stop offset="0" stopColor="#37d7fa" />
          <stop offset="0.4" stopColor="#4b72fe" />
          <stop offset="0.68" stopColor="#ff8df2" />
          <stop offset="1" stopColor="#ff8705" />
        </linearGradient>
      </defs>
      {/* Measured: the half you already know. */}
      <path
        d="M16 7 L8.2 11.5 L8.2 20.5 L16 25"
        fill="none"
        stroke={`url(#${id})`}
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Predicted: the half a model supplied. */}
      <path
        d="M16 25 L23.8 20.5 L23.8 11.5 L16 7"
        fill="none"
        stroke={`url(#${id})`}
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray="3.2 2.6"
      />
    </svg>
  );
}
