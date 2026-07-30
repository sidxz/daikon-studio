"use client";

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
        // only onSuccess invalidations. A hook writes its own onError solely
        // when the failure needs more than a message -- the dataset wizard's
        // 422, which renders a whole ValidationReport, is the one such case.
        mutationCache: new MutationCache({
          onError: (error) => {
            showError(error instanceof Error ? error.message : "Operation failed");
          },
        }),
      }),
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
