"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import type { Condition } from "@/features/engines";
import { PINNED_BY_PRETRAINED, enginesForTargetKind, useEngines } from "@/features/engines";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/shared/components/ui/collapsible";
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
import { ChevronDownIcon } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { isTerminal, useRunPoll, useTrainProtocol } from "../hooks/use-protocols";
import { ConditionFields } from "./condition-fields";

/**
 * The two fields `resolveConditions` needs, projected from the generated
 * `Condition` manifest type rather than redeclared -- a real engine's full
 * `Condition[]` satisfies this, and so does a test's minimal fixture.
 */
type ConditionDefault = Pick<Condition, "key" | "default">;

/**
 * Conditions as the server will see them: form state over manifest defaults.
 * Comparing raw form state instead would call `{n_estimators: 500}` different
 * from `{}` even though `validate_conditions` resolves both to the same dict.
 */
export function resolveConditions(
  specs: ConditionDefault[],
  values: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(specs.map((spec) => [spec.key, values[spec.key] ?? spec.default]));
}

/** The client-side mirror of the server's `baseline_is_self`. */
export function comparesAgainstItself(
  engineId: string,
  conditions: Record<string, unknown>,
  baselineEngineId: string,
  baselineConditions: Record<string, unknown>,
  specs: ConditionDefault[],
  baselineSpecs: ConditionDefault[],
): boolean {
  if (!engineId || engineId !== baselineEngineId) return false;
  return (
    JSON.stringify(resolveConditions(specs, conditions)) ===
    JSON.stringify(resolveConditions(baselineSpecs, baselineConditions))
  );
}

export function TrainProtocolForm() {
  const router = useRouter();
  const params = useSearchParams();

  const [datasetId, setDatasetId] = useState(params.get("dataset") ?? "");
  const [engineId, setEngineId] = useState("");
  const [name, setName] = useState("");
  const [conditions, setConditions] = useState<Record<string, unknown>>({});
  const [baselineEngineId, setBaselineEngineId] = useState("");
  const [baselineConditions, setBaselineConditions] = useState<Record<string, unknown>>({});
  const [runId, setRunId] = useState<string | undefined>();

  const datasets = useDatasets(undefined, 200);
  const engines = useEngines();
  const { data: dataset } = useDataset(datasetId || undefined);
  const train = useTrainProtocol();
  const run = useRunPoll(runId);

  // Only engines that can learn this dataset's kind of target. The two
  // vocabularies differ, and that translation lives in the engines feature.
  const eligible =
    engines.data && dataset ? enginesForTargetKind(engines.data, dataset.target.kind) : [];

  // Reset the engine -- and the baseline alongside it -- when the dataset
  // changes to one either cannot handle, rather than silently submitting an
  // impossible pair or a stale baseline left over from the last dataset.
  useEffect(() => {
    if (engineId && eligible.length > 0 && !eligible.some((e) => e.id === engineId)) {
      setEngineId("");
      setConditions({});
    }
    if (
      baselineEngineId &&
      eligible.length > 0 &&
      !eligible.some((e) => e.id === baselineEngineId)
    ) {
      setBaselineEngineId("");
      setBaselineConditions({});
    }
  }, [engineId, baselineEngineId, eligible]);

  const engine = eligible.find((candidate) => candidate.id === engineId);

  // A baseline comparison is mandatory, so the form defaults to the manifest's
  // flagged engine the moment there is something to default to -- the user
  // chooses which baseline, never whether to have one.
  useEffect(() => {
    if (!baselineEngineId && eligible.length > 0) {
      const flagged = eligible.find((candidate) => candidate.is_baseline);
      if (flagged) setBaselineEngineId(flagged.id);
    }
  }, [baselineEngineId, eligible]);

  const baselineEngine = eligible.find((candidate) => candidate.id === baselineEngineId);

  // What CheMeleon's checkpoint fixes for each side, so the settings block can
  // render them disabled with their true value instead of the stale default.
  const pinned = PINNED_BY_PRETRAINED[String(conditions.pretrained ?? "none")] ?? {};
  const baselinePinned =
    PINNED_BY_PRETRAINED[String(baselineConditions.pretrained ?? "none")] ?? {};

  // Keep the submitted record honest with what's displayed: the moment
  // `pretrained` picks a weight set that fixes settings, merge them into form
  // state so Train posts the value that actually ran, not whatever the input
  // was left showing.
  useEffect(() => {
    const toPin = PINNED_BY_PRETRAINED[String(conditions.pretrained ?? "none")] ?? {};
    if (Object.keys(toPin).length === 0) return;
    setConditions((prev) => ({ ...prev, ...toPin }));
  }, [conditions.pretrained]);

  useEffect(() => {
    const toPin = PINNED_BY_PRETRAINED[String(baselineConditions.pretrained ?? "none")] ?? {};
    if (Object.keys(toPin).length === 0) return;
    setBaselineConditions((prev) => ({ ...prev, ...toPin }));
  }, [baselineConditions.pretrained]);

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
        baseline_engine_id: baselineEngineId,
        baseline_conditions: baselineConditions,
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

  // The client-side mirror of the server's `baseline_is_self` -- computed once
  // here so both the submit gate and the warning below agree on it.
  const selfCompare = comparesAgainstItself(
    engineId,
    conditions,
    baselineEngineId,
    baselineConditions,
    engine?.conditions ?? [],
    baselineEngine?.conditions ?? [],
  );

  const working = train.isPending || Boolean(runId);
  // Blocked when the run would silently compare an engine against itself --
  // one fit runs and its metrics get reported as both sides of a comparison
  // that never happened. Exempt only the registry's own flagged baseline
  // engine, for which there truly is nothing else to compare against.
  const canSubmit =
    Boolean(name.trim() && datasetId && engineId && baselineEngineId) &&
    (!selfCompare || Boolean(engine?.is_baseline)) &&
    !working;

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
          A protocol is a trained model someone else can run. It is always scored against a baseline
          — one is chosen for you, and you can change it, but you cannot skip the comparison.
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
            <Select
              value={engineId}
              onValueChange={(value) => {
                setEngineId(value);
                // Otherwise a chemprop condition like `depth` lingers in state
                // and, if the new engine is ECFP4, fails validation server-side
                // as an unknown condition -- a run that never gets to fit.
                setConditions({});
              }}
              disabled={!dataset}
            >
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
          </div>

          <div className="space-y-1.5">
            <Label>Compare against</Label>
            <Select
              value={baselineEngineId}
              onValueChange={(value) => {
                setBaselineEngineId(value);
                // Same reason as the chosen-engine selector above: stale
                // conditions from the previous baseline engine otherwise
                // survive the switch and fail validation on submit.
                setBaselineConditions({});
              }}
              disabled={!dataset}
            >
              <SelectTrigger>
                <SelectValue placeholder={dataset ? "Choose a baseline" : "Pick a dataset first"} />
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
            {selfCompare && (
              <p className="text-xs text-muted-foreground">
                This is the same engine with the same settings on both sides, so there is nothing to
                compare. Change a setting on one side, or pick a different engine to measure
                against.
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
                pinned={pinned}
              />
            </div>
          )}

          {baselineEngine && (
            <Collapsible className="border-t pt-4">
              <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                  {baselineEngine.name} baseline settings
                </p>
                <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground" />
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3">
                <ConditionFields
                  conditions={baselineEngine.conditions}
                  values={baselineConditions}
                  onChange={(key, value) =>
                    setBaselineConditions((prev) => ({ ...prev, [key]: value }))
                  }
                  pinned={baselinePinned}
                />
              </CollapsibleContent>
            </Collapsible>
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
