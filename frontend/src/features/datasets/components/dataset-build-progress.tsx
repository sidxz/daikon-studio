"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Progress } from "@/shared/components/ui/progress";
import type { DatasetBuildResponse } from "@/shared/lib/api/model";
import { useEffect, useState } from "react";
import { elapsedLabel } from "./profile-computing";

/** Whole percent of `done` out of `total`, or null for a stage with no row count. */
export function buildPercent(build: Pick<DatasetBuildResponse, "done" | "total">): number | null {
  return build.total > 0 ? Math.min(100, Math.floor((build.done / build.total) * 100)) : null;
}

/**
 * What the scientist sees while a dataset builds: the stage the server reports,
 * and for a stage that counts rows, a bar and the count itself. A stage without a
 * count (reading the file, saving) shows no bar rather than an invented one.
 */
export function DatasetBuildProgress({ build }: { build: DatasetBuildResponse }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const percent = buildPercent(build);

  return (
    <Card className="mx-auto w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="text-base">Building {build.name}</CardTitle>
        <p className="text-sm text-muted-foreground">
          Every structure is checked and standardized, which takes a few minutes for a large file.
          The build continues if you leave this page; the dataset appears in the list when it is
          ready.
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-baseline justify-between text-sm">
          <span className="font-medium">{build.stage}</span>
          {percent !== null && <span className="tabular-nums">{percent}%</span>}
        </div>
        {percent !== null && <Progress value={percent} aria-label={build.stage} />}
        <p className="text-xs tabular-nums text-muted-foreground">
          {percent !== null &&
            `${build.done.toLocaleString()} of ${build.total.toLocaleString()} structures · `}
          {elapsedLabel(Date.parse(build.created_at), now)} elapsed
        </p>
      </CardContent>
    </Card>
  );
}
