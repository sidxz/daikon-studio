import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    // Playwright specs are `.spec.ts` too, and vitest would happily collect
    // them and then fail on `test.beforeEach` from a different runner. The two
    // suites stay in their own lanes: `pnpm test` is the fast unit loop,
    // `pnpm test:e2e` drives a browser.
    exclude: ["**/node_modules/**", "**/dist/**", "tests/e2e/**"],
  },
});
