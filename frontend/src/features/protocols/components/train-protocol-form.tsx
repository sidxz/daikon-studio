"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import type { Condition } from "@/features/engines";
import {
  EngineExplainer,
  PINNED_BY_PRETRAINED,
  TargetsHint,
  enginesForTargets,
  tasksForTargets,
  trainingKind,
  useEngines,
} from "@/features/engines";
// Direct, not through "@/features/runs": that index imports this feature back.
import { TrainingProgress } from "@/features/runs/components/training-progress";
import { useRunEpochs } from "@/features/runs/hooks/use-runs";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
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
import { ApiError } from "@/shared/lib/api/custom-instance";
import { isTerminal } from "@/shared/lib/query-defaults";
import { showError } from "@/shared/lib/toast";
import { ChevronDownIcon } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useRunPoll, useTrainProtocol } from "../hooks/use-protocols";
import { conditionsValid, withoutInapplicable } from "../lib/conditions";
import { ConditionFields } from "./condition-fields";
import { TuneCutoffsField } from "./tune-cutoffs-field";

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
  const [tuneCutoffs, setTuneCutoffs] = useState(false);
  const [runId, setRunId] = useState<string | undefined>();

  const datasets = useDatasets(undefined, 200);
  const engines = useEngines();
  const { data: dataset } = useDataset(datasetId || undefined);
  const train = useTrainProtocol();
  const run = useRunPoll(runId);
  const epochs = useRunEpochs(runId ?? "", Boolean(runId));

  // Only engines that can learn every one of this dataset's targets. The two
  // vocabularies differ, and that translation lives in the engines feature.
  const eligible = engines.data && dataset ? enginesForTargets(engines.data, dataset.targets) : [];

  // Unknown until a dataset is chosen: every setting shows, and the cutoff
  // option stays hidden because there is no active/inactive target to tune.
  const tasks = dataset ? tasksForTargets(dataset.targets) : undefined;
  const canTuneCutoffs = Boolean(tasks?.includes("binary_classification"));

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
    if (run.data?.status === "cancelled") {
      showError("Training was canceled");
      setRunId(undefined);
    }
  }, [run.data, router]);

  // Polling stops when the request fails with nothing loaded (`pollInterval`),
  // so without this the form waits on a bar that never moves. Only a loading
  // error: a failed *refetch* with the run already in hand is a blip the poll
  // backs off through. Fires once: with the run id cleared the hook watches no
  // query. A silent 401 belongs to the session renewal, which reloads the page.
  useEffect(() => {
    if (!run.isLoadingError) return;
    if (!(run.error instanceof ApiError && run.error.silent)) {
      showError(
        "Could not retrieve the status of this training run. It may still complete; check Protocols.",
      );
    }
    setRunId(undefined);
  }, [run.isLoadingError, run.error]);

  async function submit() {
    try {
      const created = await train.mutateAsync({
        name: name.trim(),
        dataset_id: datasetId,
        engine_id: engineId,
        // Settings the form hides for this dataset are not sent. The server resets
        // them anyway; this keeps the request what the scientist saw.
        conditions: withoutInapplicable(engine?.conditions ?? [], conditions, tasks),
        baseline_engine_id: baselineEngineId,
        baseline_conditions: withoutInapplicable(
          baselineEngine?.conditions ?? [],
          baselineConditions,
          tasks,
        ),
        // Not just the checkbox: it may have been ticked before the dataset
        // changed to one with no active/inactive target.
        tune_cutoffs: canTuneCutoffs && tuneCutoffs,
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
  // A hidden setting does not count: the server resets it before comparing.
  const selfCompare = comparesAgainstItself(
    engineId,
    withoutInapplicable(engine?.conditions ?? [], conditions, tasks),
    baselineEngineId,
    withoutInapplicable(baselineEngine?.conditions ?? [], baselineConditions, tasks),
    engine?.conditions ?? [],
    baselineEngine?.conditions ?? [],
  );

  const working = train.isPending || Boolean(runId);
  // Blocked when the run would silently compare an engine against itself --
  // one fit runs and its metrics get reported as both sides of a comparison
  // that never happened. Exempt only the registry's own flagged baseline
  // engine, for which there truly is nothing else to compare against.
  // A value outside a setting's bounds fails the run in the worker, so it blocks
  // here instead. The baseline's settings sit in a collapsed section, which is why
  // the reason is also shown beside the button.
  const settingsValid =
    conditionsValid(engine?.conditions ?? [], conditions, tasks, pinned) &&
    conditionsValid(baselineEngine?.conditions ?? [], baselineConditions, tasks, baselinePinned);
  const canSubmit =
    Boolean(name.trim() && datasetId && engineId && baselineEngineId) &&
    (!selfCompare || Boolean(engine?.is_baseline)) &&
    settingsValid &&
    !working;

  if (working && epochs.data && epochs.data.length > 0) {
    // A neural fit reports each epoch: the same live charts as the run's own page.
    return (
      <div className="w-full min-w-0 space-y-4">
        <PageHeader title={`Training ${name}`} />
        <TrainingProgress
          points={epochs.data}
          live
          phase={run.data?.phase}
          progress={run.data?.progress}
        />
      </div>
    );
  }

  if (working) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <PageHeader title={`Training ${name}`} />
        <Card className="w-full max-w-3xl">
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
            <p className="text-sm text-muted-foreground">
              You can leave this page. Training continues, and it is listed under Protocols until it
              finishes.
            </p>
            <p className="text-xs text-muted-foreground">
              Training the selected engine and the baseline. On a scaffold split, the engine is also
              trained on a random split to measure the optimism gap.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Train a protocol"
        description="Train a model and compare it with a baseline. Review its scorecard before publishing it for others to run."
      />

      <div className="w-full min-w-0 max-w-3xl space-y-4">
        {datasets.isError && (
          <QueryError
            title="Could not load datasets"
            retry={() => datasets.refetch()}
            retrying={datasets.isFetching}
          />
        )}
        {engines.isError && (
          <QueryError
            title="Could not load engines"
            retry={() => engines.refetch()}
            retrying={engines.isFetching}
          />
        )}
        {datasets.data && datasets.data.items.length === 0 && (
          <div className="rounded-lg border border-dashed bg-card p-6 text-center">
            <p className="font-medium">Create a dataset before training</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload structures and measurements to give your model something to learn from.
            </p>
            <Button asChild className="mt-4">
              <Link href="/datasets/new">Create a dataset</Link>
            </Button>
          </div>
        )}

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
              {dataset && <TargetsHint dataset={dataset} engines={engines.data ?? []} />}
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
                  <SelectValue
                    placeholder={dataset ? "Choose an engine" : "Choose a dataset first"}
                  />
                </SelectTrigger>
                <SelectContent>
                  {eligible.map((candidate) => (
                    <SelectItem key={candidate.id} value={candidate.id}>
                      {candidate.name}
                      {candidate.is_baseline ? " · default baseline" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {engine && <EngineExplainer engineId={engine.id} />}
              {engine && dataset && trainingKind(engine, dataset.targets.length) && (
                <p className="text-xs text-muted-foreground">
                  {trainingKind(engine, dataset.targets.length)}
                </p>
              )}
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
                  <SelectValue
                    placeholder={dataset ? "Choose a baseline" : "Choose a dataset first"}
                  />
                </SelectTrigger>
                <SelectContent>
                  {eligible.map((candidate) => (
                    <SelectItem key={candidate.id} value={candidate.id}>
                      {candidate.name}
                      {candidate.is_baseline ? " · default baseline" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selfCompare && (
                <p className="text-xs text-muted-foreground">
                  The model and baseline use the same engine and settings. Change a setting or
                  choose a different baseline engine.
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

            {canTuneCutoffs && <TuneCutoffsField checked={tuneCutoffs} onChange={setTuneCutoffs} />}

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
                  tasks={tasks}
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
                    tasks={tasks}
                    idPrefix="baseline-condition"
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
          <div className="flex items-center gap-3">
            {!settingsValid && (
              <span className="text-sm text-destructive">
                A setting is outside its allowed range.
              </span>
            )}
            <Button onClick={submit} disabled={!canSubmit}>
              Train
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
