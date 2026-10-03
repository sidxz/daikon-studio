"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import { enginesForTargetKind, useEngines } from "@/features/engines";
import { ConditionFields, resolveConditions } from "@/features/protocols";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent } from "@/shared/components/ui/card";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { XIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useSubmitSweep } from "../hooks/use-sweeps";

interface ConfigRow {
  id: string;
  engineId: string;
  conditions: Record<string, unknown>;
}

function emptyRow(): ConfigRow {
  return { id: crypto.randomUUID(), engineId: "", conditions: {} };
}

export function SweepForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [baselineEngineId, setBaselineEngineId] = useState("");
  const [baselineConditions, setBaselineConditions] = useState<Record<string, unknown>>({});
  const [configs, setConfigs] = useState<ConfigRow[]>(() => [emptyRow()]);

  const datasets = useDatasets(undefined, 200);
  const engines = useEngines();
  const { data: dataset } = useDataset(datasetId || undefined);
  const submit = useSubmitSweep();

  // Every row and the baseline pick from the same dataset, so the eligible
  // list is computed once and shared -- only the resolved *specs* are per-row.
  const available = dataset ? enginesForTargetKind(engines.data ?? [], dataset.target.kind) : [];
  const specsFor = (engineId: string) =>
    available.find((engine) => engine.id === engineId)?.conditions ?? [];

  function updateConfig(index: number, patch: Partial<ConfigRow>) {
    setConfigs((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  async function onSubmit() {
    try {
      const response = await submit.mutateAsync({
        name,
        dataset_id: datasetId,
        baseline_engine_id: baselineEngineId || null,
        baseline_conditions: resolveConditions(specsFor(baselineEngineId), baselineConditions),
        // Resolved against manifest defaults, exactly as the train form does.
        // Sending raw form state would make `{}` and `{n_estimators: 500}` two
        // different requests for work the server resolves identically -- which
        // would defeat the cache key and make two identical fits look like two
        // different sweep members.
        configs: configs.map((row) => ({
          engine_id: row.engineId,
          conditions: resolveConditions(specsFor(row.engineId), row.conditions),
        })),
      });
      router.push(`/sweeps/${response.sweep_id}`);
    } catch {
      // The global mutation handler has already surfaced the message.
    }
  }

  return (
    <form
      className="mx-auto w-full max-w-2xl space-y-4 p-2"
      onSubmit={(event) => {
        event.preventDefault();
        void onSubmit();
      }}
    >
      <div>
        <h1 className="text-lg font-semibold">New sweep</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Train several engine configurations on one dataset and rank them by primary metric.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-4 py-6">
          <div className="space-y-1.5">
            <Label>Dataset</Label>
            {datasets.isLoading ? (
              <Skeleton className="h-9 w-full" />
            ) : (
              <Select value={datasetId} onValueChange={setDatasetId}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a dataset" />
                </SelectTrigger>
                <SelectContent>
                  {(datasets.data?.items ?? []).map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.name} · {item.row_count.toLocaleString()} compounds
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {dataset && (
              <p className="text-xs text-muted-foreground">
                Predicting <span className="font-mono">{dataset.target.column}</span>
                {dataset.target.unit && (
                  <>
                    {" "}
                    in <span className="font-mono">{dataset.target.unit}</span>
                  </>
                )}
                , held out by {dataset.split.strategy} split.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="sweep-name">Name</Label>
            <Input
              id="sweep-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="e.g. Solubility — engine sweep"
            />
          </div>

          <div className="space-y-1.5 border-t pt-4">
            <Label>Baseline (optional)</Label>
            <Select
              value={baselineEngineId}
              onValueChange={(value) => {
                setBaselineEngineId(value);
                // Otherwise a stale condition from the previous baseline
                // engine survives the switch and fails validation on submit.
                setBaselineConditions({});
              }}
              disabled={!dataset}
            >
              <SelectTrigger>
                <SelectValue placeholder={dataset ? "No baseline" : "Choose a dataset first"} />
              </SelectTrigger>
              <SelectContent>
                {available.map((candidate) => (
                  <SelectItem key={candidate.id} value={candidate.id}>
                    {candidate.name}
                    {candidate.is_baseline ? " · default baseline" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {baselineEngineId && (
              <ConditionFields
                conditions={specsFor(baselineEngineId)}
                values={baselineConditions}
                onChange={(key, value) =>
                  setBaselineConditions((prev) => ({ ...prev, [key]: value }))
                }
              />
            )}
          </div>
        </CardContent>
      </Card>

      <div className="space-y-3">
        {configs.map((row, index) => (
          <Card key={row.id} data-testid="sweep-config-row">
            <CardContent className="space-y-4 py-6">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                  Configuration {index + 1}
                </p>
                {configs.length > 1 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Remove configuration ${index + 1}`}
                    onClick={() => setConfigs((rows) => rows.filter((_, i) => i !== index))}
                  >
                    <XIcon className="size-4" aria-hidden />
                  </Button>
                )}
              </div>

              <div className="space-y-1.5">
                <Label>Engine</Label>
                <Select
                  value={row.engineId}
                  onValueChange={(value) =>
                    updateConfig(index, { engineId: value, conditions: {} })
                  }
                  disabled={!dataset}
                >
                  <SelectTrigger>
                    <SelectValue
                      placeholder={dataset ? "Choose an engine" : "Choose a dataset first"}
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {available.map((candidate) => (
                      <SelectItem key={candidate.id} value={candidate.id}>
                        {candidate.name}
                        {candidate.is_baseline ? " · default baseline" : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {row.engineId && (
                <ConditionFields
                  conditions={specsFor(row.engineId)}
                  values={row.conditions}
                  onChange={(key, value) =>
                    updateConfig(index, { conditions: { ...row.conditions, [key]: value } })
                  }
                />
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Button
        type="button"
        variant="outline"
        onClick={() => setConfigs((rows) => [...rows, emptyRow()])}
      >
        Add configuration
      </Button>

      <div className="flex items-center justify-between">
        <Button type="button" variant="ghost" onClick={() => router.push("/sweeps")}>
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={
            submit.isPending || !datasetId || !name || !configs.every((row) => row.engineId)
          }
        >
          Start sweep
        </Button>
      </div>
    </form>
  );
}
