"use client";

import { ApiError } from "@/shared/lib/api/custom-instance";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { showError } from "@/shared/lib/toast";
import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: STALE_TIME.DEFAULT, retry: 1 },
        },
        // One place turns a failed mutation into a toast, so feature hooks add
        // only onSuccess invalidations. A mutation whose failure needs more
        // than a message -- the dataset wizard's 422, which renders a whole
        // ValidationReport, or publish's 423 -- reports it itself and sets
        // `meta: { silent: true }`, so a failure is one toast, never two. A
        // silent ApiError is a 401 the session renewal is already handling.
        mutationCache: new MutationCache({
          onError: (error, _variables, _context, mutation) => {
            if (mutation.meta?.silent) return;
            if (error instanceof ApiError && error.silent) return;
            showError(error instanceof Error ? error.message : "Operation failed");
          },
        }),
      }),
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
