import { defineConfig } from "orval";

/**
 * Reads the committed `openapi.json` snapshot, not a live server. Codegen then
 * works offline and in CI, and a contract change shows up as a reviewable diff
 * rather than appearing silently the next time someone happens to have the
 * backend running. Refresh it with `make generate-api` from the repo root.
 *
 * Generates types only in practice: the hooks in `src/features/*` are
 * hand-written against `customInstance`. The triage grid's datasource is an
 * imperative `async (startRow) => ...` callback that cannot be a React hook at
 * all, and the 422 validation-report path needs handling the snapshot does not
 * describe -- so generated hooks would be dead weight either way. Import the
 * generated *type*; never hand-roll one that mirrors a backend DTO.
 */
export default defineConfig({
  studio: {
    input: { target: "./openapi.json" },
    output: {
      target: "src/shared/lib/api/endpoints.ts",
      schemas: "src/shared/lib/api/model",
      client: "react-query",
      mode: "tags-split",
      override: {
        mutator: {
          path: "src/shared/lib/api/custom-instance.ts",
          name: "customInstance",
        },
      },
    },
  },
});
