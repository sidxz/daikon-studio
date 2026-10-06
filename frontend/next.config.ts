import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Browser tests run beside the developer's server without sharing its lock or build output.
  distDir: process.env.APP_E2E === "1" ? ".next-e2e" : ".next",
  turbopack: {
    resolveAlias: {
      // RDKit.js WASM has `require('fs')` in its Node.js detection path.
      // Stub it out for client bundles -- the WASM loader uses fetch, not fs.
      fs: "./src/shared/lib/rdkit/empty-module.ts",
    },
  },
};

export default nextConfig;
