"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import { enginesForTargetKind, useEngines } from "@/features/engines";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
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
import { showError } from "@/shared/lib/toast";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { isTerminal, useRunPoll, useTrainProtocol } from "../hooks/use-protocols";
import { ConditionFields } from "./condition-fields";

export function TrainProtocolForm() {
  const router = useRouter();
  const params = useSearchParams();

  const [datasetId, setDatasetId] = useState(params.get("dataset") ?? "");
  const [engineId, setEngineId] = useState("");
  const [name, setName] = useState("");
  const [conditions, setConditions] = useState<Record<string, unknown>>({});
  const [runId, setRunId] = useState<string | undefined>();

  const datasets = useDatasets();
  const engines = useEngines();
  const { data: dataset } = useDataset(datasetId || undefined);
  const train = useTrainProtocol();
  const run = useRunPoll(runId);

  // Only engines that can learn this dataset's kind of target. The two
  // vocabularies differ, and that translation lives in the engines feature.
  const eligible =
    engines.data && dataset ? enginesForTargetKind(engines.data, dataset.target.kind) : [];

  // Reset the engine when the dataset changes to one it cannot handle,
  // rather than silently submitting an impossible pair.
  useEffect(() => {
    if (engineId && eligible.length > 0 && !eligible.some((e) => e.id === engineId)) {
      setEngineId("");
      setConditions({});
    }
  }, [engineId, eligible]);

  const engine = eligible.find((candidate) => candidate.id === engineId);

  // The Protocol does not exist until training finishes, so the run carries the
  // id back. Before Run gained that column this transition was a dead end.
  useEffect(() => {
    if (run.data?.status === "ready" && run.data.protocol_id) {
      router.push(`/protocols/${run.data.protocol_id}`);
    }
    if (run.data?.status === "failed") {
      showError(run.data.error_message ?? "Training failed");
      setRunId(undefined);
    }
  }, [run.data, router]);

  async function submit() {
    try {
      const created = await train.mutateAsync({
        name: name.trim(),
        dataset_id: datasetId,
        engine_id: engineId,
        conditions,
      });
      // A cache hit returns 202 with an already-ready Run, so branch on
      // status rather than assuming 202 means work started.
      if (isTerminal(created.status) && created.protocol_id) {
        router.push(`/protocols/${created.protocol_id}`);
        return;
      }
      setRunId(created.id);
    } catch {
      // The global mutation handler has already surfaced the message.
    }
  }

  const working = train.isPending || Boolean(runId);
  const canSubmit = Boolean(name.trim() && datasetId && engineId) && !working;

  if (working) {
    return (
      <div className="mx-auto w-full max-w-2xl p-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training {name}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">{run.data?.phase ?? "Starting…"}</p>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full bg-primary transition-all duration-500"
                style={{ width: `${Math.round((run.data?.progress ?? 0) * 100)}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Your chosen engine, the mandatory baseline, and — on a scaffold split — the same
              engine on a random split, so the optimism gap is measured rather than guessed.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-2xl space-y-4 p-2">
      <div>
        <h1 className="text-lg font-semibold">Train a protocol</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          A protocol is a trained model someone else can run. It is scored against a fingerprint
          baseline automatically — you do not get to skip that comparison.
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
            <Label>Engine</Label>
            <Select value={engineId} onValueChange={setEngineId} disabled={!dataset}>
              <SelectTrigger>
                <SelectValue placeholder={dataset ? "Choose an engine" : "Pick a dataset first"} />
              </SelectTrigger>
              <SelectContent>
                {eligible.map((candidate) => (
                  <SelectItem key={candidate.id} value={candidate.id}>
                    {candidate.name}
                    {candidate.is_baseline ? " · the baseline" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {engine?.is_baseline && (
              <p className="text-xs text-muted-foreground">
                This engine is the baseline, so there is nothing to compare it against. Its
                scorecard will say so rather than showing a comparison that never happened.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="protocol-name">Name</Label>
            <Input
              id="protocol-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="e.g. Solubility — XGBoost"
            />
          </div>

          {engine && (
            <div className="border-t pt-4">
              <p className="mb-3 text-xs font-medium uppercase tracking-widest text-muted-foreground">
                {engine.name} settings
              </p>
              <ConditionFields
                conditions={engine.conditions}
                values={conditions}
                onChange={(key, value) => setConditions((prev) => ({ ...prev, [key]: value }))}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={() => router.push("/protocols")}>
          Cancel
        </Button>
        <Button onClick={submit} disabled={!canSubmit}>
          Train
        </Button>
      </div>
    </div>
  );
}
