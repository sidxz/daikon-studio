"use client";

const MOLSTAR_VERSION = "3.3.0";
const MOLSTAR_JS = `https://cdn.jsdelivr.net/npm/pdbe-molstar@${MOLSTAR_VERSION}/build/pdbe-molstar-component.js`;
const MOLSTAR_CSS = `https://cdn.jsdelivr.net/npm/pdbe-molstar@${MOLSTAR_VERSION}/build/pdbe-molstar.css`;

let molstarPromise: Promise<void> | null = null;

/**
 * Inject the pdbe-molstar web component bundle (CDN script + stylesheet)
 * once; every caller — the first and every later one — gets the same
 * in-flight/resolved promise. Mirrors ProtCellar's structure-viewer-card.tsx
 * loader exactly. CDN delivery (not the npm `molstar` package) is deliberate:
 * a proven zero-bundler-config path for the sizeable Mol* build, same
 * rationale as RDKit-JS being a WASM npm dep instead (lib/rdkit/loader.ts) —
 * different distribution shape per library, not an inconsistency.
 */
export function loadMolstar(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.customElements?.get("pdbe-molstar")) return Promise.resolve();
  if (molstarPromise) return molstarPromise;
  molstarPromise = new Promise<void>((resolve, reject) => {
    if (!document.querySelector("link[data-molstar]")) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = MOLSTAR_CSS;
      link.dataset.molstar = "1";
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.src = MOLSTAR_JS;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load pdbe-molstar"));
    document.body.appendChild(script);
  });
  // Reset on failure so a transient CDN blip doesn't poison every later caller
  // for the whole session — the next loadMolstar() re-injects and retries.
  molstarPromise.catch(() => {
    molstarPromise = null;
  });
  return molstarPromise;
}
