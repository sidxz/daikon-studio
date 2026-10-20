"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/shared/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { ArrowUpRight, Cpu, Layers } from "lucide-react";
import { type Engine, TASK_LABELS } from "../types";
import { ConditionSummary } from "./condition-summary";
import { EngineExplainer } from "./engine-explainer";

const INPUT_LABELS: Record<string, string> = {
  molecule: "Small molecules",
  sequence: "Proteins",
};

export function computeLabel(lane: string): string {
  return lane === "gpu" ? "GPU lane" : lane === "default" ? "CPU lane" : `${lane} lane`;
}

function baselineLabel(engine: Engine): string {
  if (engine.structure_kinds.length !== 1) return "Default baseline";
  return engine.structure_kinds[0] === "molecule"
    ? "Small-molecule baseline"
    : engine.structure_kinds[0] === "sequence"
      ? "Protein baseline"
      : "Default baseline";
}

export function EngineCard({ engine }: { engine: Engine }) {
  const inputs = engine.structure_kinds.map((kind) => INPUT_LABELS[kind] ?? kind).join(" · ");

  return (
    <Sheet>
      <Card className="h-full gap-4 pb-0">
        <CardHeader className="px-5">
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span>{inputs}</span>
            <span className="inline-flex items-center gap-1.5">
              <Cpu className="size-3.5" aria-hidden="true" />
              {computeLabel(engine.lane)}
            </span>
          </div>
          <CardTitle>
            <h2 className="text-base">{engine.name}</h2>
          </CardTitle>
          <p className="line-clamp-3 text-sm leading-relaxed text-muted-foreground">
            {engine.description}
          </p>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {engine.tasks.map((task) => (
              <Badge key={task} variant="outline" className="font-normal">
                {TASK_LABELS[task] ?? task}
              </Badge>
            ))}
            {engine.is_baseline && <Badge variant="secondary">{baselineLabel(engine)}</Badge>}
          </div>
        </CardHeader>
        <CardContent className="px-5">
          <EngineExplainer engineId={engine.id} />
        </CardContent>
        <div className="mt-auto flex flex-wrap items-center justify-between gap-2 border-t px-5 py-3">
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <Layers className="size-3.5" aria-hidden="true" />
            {engine.supports_multitask ? "Joint multi-target model" : "One model per target"}
          </span>
          <SheetTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="-mr-2 text-primary"
              aria-label={`View details for ${engine.name}`}
            >
              View details
              <ArrowUpRight aria-hidden="true" />
            </Button>
          </SheetTrigger>
        </div>
      </Card>

      <SheetContent className="w-full gap-0 overflow-y-auto sm:max-w-2xl">
        <SheetHeader className="border-b p-6 pr-12">
          <p className="text-xs text-muted-foreground">{inputs}</p>
          <SheetTitle className="text-xl">{engine.name}</SheetTitle>
          <SheetDescription>Capabilities and training settings</SheetDescription>
        </SheetHeader>
        <Tabs defaultValue="overview" className="gap-5 p-6">
          <TabsList aria-label="Engine details">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="settings">Settings ({engine.conditions.length})</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="space-y-6">
            <p className="text-sm leading-relaxed text-muted-foreground">{engine.description}</p>
            {engine.is_baseline && <Badge variant="secondary">{baselineLabel(engine)}</Badge>}
            <dl className="grid grid-cols-1 gap-5 rounded-sm border bg-muted/20 p-4 text-sm sm:grid-cols-2">
              <div className="space-y-1">
                <dt className="text-xs text-muted-foreground">Input</dt>
                <dd>{inputs}</dd>
              </div>
              <div className="space-y-1">
                <dt className="text-xs text-muted-foreground">Compute</dt>
                <dd>{computeLabel(engine.lane)}</dd>
              </div>
              <div className="space-y-1">
                <dt className="text-xs text-muted-foreground">Supported tasks</dt>
                <dd>{engine.tasks.map((task) => TASK_LABELS[task] ?? task).join(" · ")}</dd>
              </div>
              <div className="space-y-1">
                <dt className="text-xs text-muted-foreground">Multiple targets</dt>
                <dd>
                  {engine.supports_multitask
                    ? "One joint model for targets of the same type"
                    : "A separate model for each target"}
                </dd>
              </div>
            </dl>
            <EngineExplainer engineId={engine.id} />
            <dl className="space-y-3 border-t pt-4 text-xs">
              <div className="flex flex-wrap justify-between gap-2">
                <dt className="text-muted-foreground">Engine ID</dt>
                <dd className="break-all font-mono">{engine.id}</dd>
              </div>
              <div className="flex flex-wrap justify-between gap-2">
                <dt className="text-muted-foreground">Version</dt>
                <dd className="font-mono">{engine.version}</dd>
              </div>
            </dl>
          </TabsContent>
          <TabsContent value="settings" className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Default values and allowed settings. Configure these when starting a training run.
            </p>
            <ConditionSummary conditions={engine.conditions} layout="table" />
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}
