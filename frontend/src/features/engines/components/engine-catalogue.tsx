"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useEngines } from "../hooks/use-engines";
import { TASK_LABELS } from "../types";
import { ConditionSummary } from "./condition-summary";
import { EngineExplainer } from "./engine-explainer";

export function EngineCatalogue() {
  const { data: engines, isLoading, isError, error } = useEngines();

  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 p-2">
      <div>
        <h1 className="text-lg font-semibold">Engines</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Engines available for training protocols, with their settings.
        </p>
      </div>

      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Skeleton className="h-56 w-full" />
          <Skeleton className="h-56 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load engines</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      )}

      {engines && (
        <div className="grid items-stretch gap-4 sm:grid-cols-2">
          {engines.map((engine) => (
            <Card key={engine.id} className="flex h-full flex-col">
              <CardHeader>
                <div className="flex items-start justify-between gap-3">
                  <CardTitle className="text-base">{engine.name}</CardTitle>
                  {engine.is_baseline && (
                    <Badge variant="secondary" className="shrink-0">
                      Default baseline
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">{engine.description}</p>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {engine.tasks.map((task) => (
                    <Badge key={task} variant="outline" className="font-normal">
                      {TASK_LABELS[task] ?? task}
                    </Badge>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="flex-1">
                <div className="mb-4">
                  <EngineExplainer engineId={engine.id} />
                </div>
                <p className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
                  Settings
                </p>
                <ConditionSummary conditions={engine.conditions} />
              </CardContent>
              <div className="border-t px-6 py-2.5">
                <span className="font-mono text-xs text-muted-foreground">
                  {engine.id} · v{engine.version}
                </span>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
