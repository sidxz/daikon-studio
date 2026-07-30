"use client";

import type { RDKitModule } from "@rdkit/rdkit";

let rdkitPromise: Promise<RDKitModule> | null = null;

/**
 * One RDKit instance for the page. The WASM binary is ~7 MB, so it is loaded
 * lazily on first use and shared thereafter -- a per-component init would
 * re-download and re-instantiate it for every rendered structure.
 *
 * The binary is copied into public/ by a postinstall hook rather than bundled,
 * which is why `locateFile` points at an absolute path.
 */
export function getRDKit(): Promise<RDKitModule> {
  if (rdkitPromise) return rdkitPromise;
  rdkitPromise = (async () => {
    const initRDKitModule = (
      (await import("@rdkit/rdkit")) as unknown as {
        default: (opts?: { locateFile?: () => string }) => Promise<RDKitModule>;
      }
    ).default;
    return await initRDKitModule({ locateFile: () => "/RDKit_minimal.wasm" });
  })();
  return rdkitPromise;
}
