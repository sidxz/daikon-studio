"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { useEffect, useState } from "react";

/** "45 s", "3 min 12 s", "1 h 2 min": how long the server has been at it. */
export function elapsedLabel(startedAt: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${seconds % 60} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/**
 * What the reader sees while the server computes a dataset's profile: what is
 * being computed, for how many compounds, and for how long -- timed from the
 * server's start, so the count survives a reload. No progress bar: the server
 * reports none, and an invented one would be a guess dressed as a measurement.
 */
export function ProfileComputing({
  startedAt,
  compounds,
}: {
  startedAt: string;
  compounds: number;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Profiling {compounds.toLocaleString()} compounds
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Computing physicochemical descriptors, Bemis–Murcko scaffolds, test-to-training
          similarity, and activity cliffs. This runs once per dataset; the result is saved, and later visits open
          instantly.
        </p>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p className="flex items-center gap-2" aria-live="polite">
          <span className="relative flex size-2" aria-hidden="true">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-60 motion-reduce:hidden" />
            <span className="relative inline-flex size-2 rounded-full bg-primary" />
          </span>
          Running for {elapsedLabel(Date.parse(startedAt), now)}. Large datasets can take several
          minutes.
        </p>
        <p className="text-muted-foreground">
          The calculation runs on the server. Leaving or reloading this page does not restart it;
          the results appear here when it finishes.
        </p>
      </CardContent>
    </Card>
  );
}
