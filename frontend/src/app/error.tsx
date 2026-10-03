"use client";

import { Button } from "@/shared/components/ui/button";

/** Any render error below the root layout lands here instead of a blank page. */
export default function RouteError({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="max-w-md space-y-3 text-center">
        <p className="text-lg font-semibold">Something went wrong</p>
        <p className="text-sm text-muted-foreground">{error.message}</p>
        <Button variant="outline" size="sm" onClick={reset}>
          Try again
        </Button>
      </div>
    </div>
  );
}
