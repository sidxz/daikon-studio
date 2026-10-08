"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import { DatasetReadinessView } from "@/features/datasets/components/dataset-readiness-view";
import { useDatasetReadiness } from "@/features/datasets/hooks/use-datasets";
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
import { Badge } from "@/shared/components/ui/badge";
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
import { hasRandomComparison, isGroupedSplit } from "@/shared/lib/split";
import { showError } from "@/shared/lib/toast";
import { Check, ChevronDownIcon, ClipboardCheck, Pencil } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useRunPoll, useTrainProtocol } from "../hooks/use-protocols";
import {
  appliesToTasks,
  conditionError,
  conditionsValid,
  withoutInapplicable,
} from "../lib/conditions";
import { DatasetPicker } from "./dataset-picker";
import { EnginePicker, computeLabel } from "./engine-picker";
import { TrainingSettings, changedSettings, settingValue } from "./training-settings";
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
  const [datasetSearch, setDatasetSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [baselineOpen, setBaselineOpen] = useState(false);
  const [revealSettings, setRevealSettings] = useState(false);
  const [selectionNotice, setSelectionNotice] = useState("");
  const lastSuggestedName = useRef("");
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(datasetSearch), 200);
    return () => window.clearTimeout(timer);
  }, [datasetSearch]);

  const datasets = useDatasets(undefined, 200, { q: debouncedSearch || undefined });
  const engines = useEngines();
  const datasetQuery = useDataset(datasetId || undefined);
  const dataset = datasetId ? datasetQuery.data : undefined;
  const readiness = useDatasetReadiness(datasetId || undefined);
  const train = useTrainProtocol();
  const run = useRunPoll(runId);
  const epochs = useRunEpochs(runId ?? "", Boolean(runId));

  // Only engines that can learn every one of this dataset's targets. The two
  // vocabularies differ, and that translation lives in the engines feature.
  const eligible =
    engines.data && dataset
      ? enginesForTargets(
          engines.data,
          dataset.targets,
          dataset.validation_report?.structure_kind ?? "molecule",
        )
      : [];

  // Unknown until a dataset is chosen: every setting shows, and the cutoff
  // option stays hidden because there is no active/inactive target to tune.
  const tasks = dataset ? tasksForTargets(dataset.targets) : undefined;
  const canTuneCutoffs = Boolean(tasks?.includes("binary_classification"));

  // Reset the engine -- and the baseline alongside it -- when the dataset
  // changes to one either cannot handle, rather than silently submitting an
  // impossible pair or a stale baseline left over from the last dataset.
  useEffect(() => {
    if (dataset && engines.data && engineId && !eligible.some((e) => e.id === engineId)) {
      setEngineId("");
      setConditions({});
      setSelectionNotice(
        "The previous engine cannot train these target types. Choose a compatible engine.",
      );
    }
    if (
      baselineEngineId &&
      dataset &&
      engines.data &&
      !eligible.some((e) => e.id === baselineEngineId)
    ) {
      setBaselineEngineId("");
      setBaselineConditions({});
    }
  }, [engineId, baselineEngineId, eligible, dataset, engines.data]);

  const engine = eligible.find((candidate) => candidate.id === engineId);
  const proposedName = dataset && engine ? `${dataset.name} · ${engine.name}` : null;
  useEffect(() => {
    if (!proposedName) return;
    const previousSuggestion = lastSuggestedName.current;
    setName((previous) => (!previous || previous === previousSuggestion ? proposedName : previous));
    lastSuggestedName.current = proposedName;
  }, [proposedName]);

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
    Boolean(name.trim() && dataset && engine && baselineEngine) &&
    (!selfCompare || Boolean(engine?.is_baseline)) &&
    settingsValid &&
    !working;

  const issues: { message: string; id: string; baseline?: boolean }[] = [];
  if (!datasetId) issues.push({ message: "Choose a dataset.", id: "training-dataset" });
  else if (datasetQuery.isError)
    issues.push({ message: "Reload the selected dataset.", id: "training-data" });
  else if (!dataset)
    issues.push({ message: "Wait for the selected dataset to load.", id: "training-data" });
  if (dataset && !engine)
    issues.push({
      message: eligible.length
        ? "Choose an engine."
        : "No compatible engine is available for these targets.",
      id: "training-engine",
    });
  if (dataset && !baselineEngine)
    issues.push({
      message: "Choose a compatible baseline.",
      id: "training-baseline",
      baseline: true,
    });
  if (!name.trim()) issues.push({ message: "Enter a protocol name.", id: "protocol-name" });
  if (selfCompare && !engine?.is_baseline)
    issues.push({
      message:
        "Choose a different baseline or change its settings; the current comparison is identical.",
      id: "training-baseline",
      baseline: true,
    });
  for (const [specs, values, fixed, baseline] of [
    [engine?.conditions ?? [], conditions, pinned, false],
    [baselineEngine?.conditions ?? [], baselineConditions, baselinePinned, true],
  ] as const) {
    for (const spec of specs) {
      if (!appliesToTasks(spec, tasks) || spec.key in fixed) continue;
      const error = conditionError(spec, values[spec.key] ?? spec.default);
      if (error)
        issues.push({
          message: `${baseline ? "Baseline: " : ""}${spec.label}: ${error}`,
          id: `${baseline ? "baseline-condition" : "condition"}-${spec.key}`,
          baseline,
        });
    }
  }
  function focusIssue(issue: { id: string; baseline?: boolean }) {
    if (issue.baseline) setBaselineOpen(true);
    setRevealSettings(true);
    window.requestAnimationFrame(() => {
      const element = document.getElementById(issue.id);
      element?.scrollIntoView({ behavior: "smooth", block: "center" });
      element?.focus({ preventScroll: true });
    });
  }

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
              Training the selected engine and the baseline. Unless the dataset is split at random,
              the engine is also trained on a random split to measure the optimism gap.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full min-w-0 space-y-6">
      <PageHeader
        title="Train a protocol"
        description="Choose your data and model, review the evaluation plan, then start training."
      />
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
      {selectionNotice && (
        <p aria-live="polite" className="rounded-lg border bg-muted/30 p-3 text-sm">
          {selectionNotice}
        </p>
      )}
      <div className="grid max-w-7xl items-start gap-5 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-w-0 space-y-5">
          <Card id="training-data" className="scroll-mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <span className="flex size-6 items-center justify-center rounded-full bg-muted text-xs">
                  1
                </span>
                Choose your dataset
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Select the structures and measurements this protocol will learn from.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="training-dataset">Dataset</Label>
                <DatasetPicker
                  items={datasets.data?.items ?? []}
                  selected={dataset}
                  search={datasetSearch}
                  onSearch={setDatasetSearch}
                  onSelect={(id) => {
                    setDatasetId(id);
                    setSelectionNotice("");
                  }}
                  loading={datasets.isFetching ?? datasets.isLoading}
                  failed={datasets.isError}
                />
              </div>
              {datasets.data?.items.length === 0 && !debouncedSearch && !datasetId && (
                <div className="rounded-lg border border-dashed p-4">
                  <p className="text-sm font-medium">Create a dataset before training</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Upload structures and measurements to give your model something to learn from.
                  </p>
                  <Button asChild variant="outline" size="sm" className="mt-3">
                    <Link href="/datasets/new">Create a dataset</Link>
                  </Button>
                </div>
              )}
              {datasetQuery.isError && (
                <QueryError
                  title="Could not load the selected dataset"
                  retry={() => datasetQuery.refetch()}
                  retrying={datasetQuery.isFetching}
                />
              )}
              {datasetId && !dataset && !datasetQuery.isError && (
                <Skeleton className="h-20 w-full" />
              )}
              {dataset && (
                <>
                  <div className="rounded-lg bg-muted/30 p-4">
                    <p className="mb-2 text-sm font-medium">
                      {dataset.row_count?.toLocaleString()} unique compounds ·{" "}
                      {dataset.targets.length} {dataset.targets.length === 1 ? "target" : "targets"}
                    </p>
                    <TargetsHint dataset={dataset} engines={engines.data ?? []} />
                    <ul className="mt-3 space-y-1">
                      {dataset.targets.map((target) => (
                        <li
                          key={target.column}
                          className="flex flex-wrap items-center gap-2 text-xs"
                        >
                          <span className="font-mono font-medium">{target.column}</span>
                          <span className="text-muted-foreground">
                            {target.kind === "binary" ? "Active / inactive" : "Measured value"}
                            {target.unit ? ` · ${target.unit}` : ""}
                            {target.direction
                              ? ` · ${target.direction === "high" ? "higher" : "lower"} is better`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  {readiness.isLoading && (
                    <p aria-live="polite" className="text-xs text-muted-foreground">
                      Loading exact partition counts and data checks…
                    </p>
                  )}
                  {readiness.isError && (
                    <QueryError
                      title="Could not load data checks"
                      retry={() => readiness.refetch()}
                      retrying={readiness.isFetching}
                    />
                  )}
                  {readiness.data && <DatasetReadinessView readiness={readiness.data} />}
                  {dataset.validation_report &&
                    (dataset.validation_report.invalid.length > 0 ||
                      dataset.validation_report.duplicates_collapsed > 0 ||
                      dataset.validation_report.conflicting.length > 0 ||
                      dataset.validation_report.salts_flagged > 0) && (
                      <p className="rounded-lg border p-3 text-xs text-muted-foreground">
                        Preparation excluded {dataset.validation_report.invalid.length} invalid rows
                        and{" "}
                        {
                          new Set(dataset.validation_report.conflicting.map((row) => row.structure))
                            .size
                        }{" "}
                        compounds with conflicting labels, merged{" "}
                        {dataset.validation_report.duplicates_collapsed} duplicate rows, and flagged{" "}
                        {dataset.validation_report.salts_flagged} salts or mixtures. See the dataset
                        report for details.
                      </p>
                    )}
                  <Button asChild variant="link" size="sm" className="h-auto px-0">
                    <Link href={`/datasets/${dataset.id}`}>View dataset and full diagnostics</Link>
                  </Button>
                </>
              )}
            </CardContent>
          </Card>

          <Card id="training-engine" tabIndex={-1} className="scroll-mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <span className="flex size-6 items-center justify-center rounded-full bg-muted text-xs">
                  2
                </span>
                Choose an engine
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Compare the models that can learn your targets. Settings start at their defaults.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {!dataset ? (
                <p className="text-sm text-muted-foreground">
                  Choose a dataset to see compatible engines.
                </p>
              ) : engines.isLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : (
                <EnginePicker
                  engines={engines.data ?? []}
                  eligible={eligible}
                  value={engineId}
                  onChange={(id) => {
                    setEngineId(id);
                    setConditions({});
                    setRevealSettings(false);
                  }}
                  targetCount={dataset.targets.length}
                  structureKind={dataset.validation_report?.structure_kind}
                />
              )}
              {engine && (
                <details className="rounded-lg border p-3">
                  <summary className="cursor-pointer text-sm font-medium">
                    How {engine.name} learns
                  </summary>
                  <div className="mt-3">
                    <EngineExplainer engineId={engine.id} />
                  </div>
                </details>
              )}
            </CardContent>
          </Card>

          <Card id="training-evaluation" className="scroll-mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <span className="flex size-6 items-center justify-center rounded-full bg-muted text-xs">
                  3
                </span>
                Evaluation and settings
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                The model and baseline use the same held-out data, with matching evaluation options.
              </p>
            </CardHeader>
            <CardContent className="space-y-5">
              {engine ? (
                <TrainingSettings
                  title={`${engine.name} settings`}
                  conditions={engine.conditions}
                  values={conditions}
                  onChange={(key, value) =>
                    setConditions((previous) => ({ ...previous, [key]: value }))
                  }
                  onReset={() => {
                    setConditions({});
                    setRevealSettings(false);
                  }}
                  pinned={pinned}
                  tasks={tasks}
                  idPrefix="condition"
                  reveal={revealSettings}
                />
              ) : (
                <p className="text-sm text-muted-foreground">
                  Choose an engine to configure its settings.
                </p>
              )}
              <Collapsible
                open={baselineOpen}
                onOpenChange={setBaselineOpen}
                className="rounded-lg border p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium">Comparison baseline</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {baselineEngine?.name ?? "Choose a dataset to select a baseline"}
                    </p>
                  </div>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Pencil className="size-3" />
                      {baselineOpen ? "Hide baseline settings" : "Change baseline"}
                      <ChevronDownIcon className="size-3" />
                    </Button>
                  </CollapsibleTrigger>
                </div>
                <p className="mt-3 text-xs text-muted-foreground">
                  A baseline comparison is included in every protocol. It shows whether your model
                  adds value over the reference.
                </p>
                <CollapsibleContent className="mt-4 space-y-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="training-baseline">Compare against</Label>
                    <Select
                      value={baselineEngineId}
                      onValueChange={(value) => {
                        setBaselineEngineId(value);
                        setBaselineConditions({});
                      }}
                      disabled={!dataset}
                    >
                      <SelectTrigger id="training-baseline">
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
                  </div>
                  {baselineEngine && (
                    <TrainingSettings
                      title={`${baselineEngine.name} baseline settings`}
                      conditions={baselineEngine.conditions}
                      values={baselineConditions}
                      onChange={(key, value) =>
                        setBaselineConditions((previous) => ({ ...previous, [key]: value }))
                      }
                      onReset={() => setBaselineConditions({})}
                      pinned={baselinePinned}
                      tasks={tasks}
                      idPrefix="baseline-condition"
                      reveal={revealSettings}
                    />
                  )}
                </CollapsibleContent>
              </Collapsible>
              {selfCompare && (
                <p
                  aria-live="polite"
                  className={`rounded-lg border p-3 text-xs ${engine?.is_baseline ? "text-muted-foreground" : "border-destructive/40 text-destructive"}`}
                >
                  {engine?.is_baseline
                    ? "The selected model is the baseline reference. One fit supplies both results; the scorecard labels this explicitly."
                    : "The model and baseline use the same engine and settings. Change a setting or choose a different baseline engine."}
                </p>
              )}
              {canTuneCutoffs && (
                <div className="rounded-lg border p-4">
                  <TuneCutoffsField checked={tuneCutoffs} onChange={setTuneCutoffs} />
                </div>
              )}
              {dataset != null && hasRandomComparison(dataset.split.strategy) && (
                <p className="rounded-lg bg-muted/30 p-3 text-xs text-muted-foreground">
                  Training also fits the selected engine on a random split, with the same seed and
                  the same partition sizes. Its score is compared with the {dataset.split.strategy}
                  -split result to measure the optimism gap.
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        <aside className="min-w-0 space-y-4" aria-label="Training plan">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <ClipboardCheck className="size-4" />
                Review and train
              </CardTitle>
              <p className="text-xs text-muted-foreground">
                Check the complete plan before starting.
              </p>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-1.5">
                <Label htmlFor="protocol-name">Name</Label>
                <Input
                  id="protocol-name"
                  value={name}
                  maxLength={256}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. Solubility · XGBoost"
                />
              </div>
              <dl className="space-y-3 text-sm">
                <div>
                  <dt className="text-xs text-muted-foreground">Dataset</dt>
                  <dd className="mt-1 font-medium">{dataset?.name ?? "Choose a dataset"}</dd>
                  {dataset && (
                    <dd className="mt-1 text-xs text-muted-foreground">
                      {dataset.row_count?.toLocaleString()} compounds · {dataset.targets.length}{" "}
                      targets
                    </dd>
                  )}
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Predictions</dt>
                  <dd className="mt-1 space-y-1">
                    {dataset
                      ? dataset.targets.map((target) => (
                          <div key={target.column} className="text-xs">
                            <span className="font-mono">{target.column}</span>
                            <span className="text-muted-foreground">
                              {" "}
                              ·{" "}
                              {target.kind === "binary"
                                ? "0 / 1"
                                : target.unit || "unit unspecified"}
                            </span>
                          </div>
                        ))
                      : "Choose targets via a dataset"}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Held-out evaluation</dt>
                  <dd className="mt-1 capitalize">
                    {dataset
                      ? `${dataset.split.strategy} split · seed ${dataset.split.seed}`
                      : "Defined by the dataset"}
                  </dd>
                  {readiness.data && (
                    <dd className="mt-1 text-xs text-muted-foreground">
                      {readiness.data.partition_counts.train?.toLocaleString()} train /{" "}
                      {readiness.data.partition_counts.validation?.toLocaleString()} validation /{" "}
                      {readiness.data.partition_counts.test?.toLocaleString()} test
                    </dd>
                  )}
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Model</dt>
                  <dd className="mt-1 font-medium">{engine?.name ?? "Choose an engine"}</dd>
                  {engine && (
                    <dd className="mt-1 text-xs text-muted-foreground">
                      {computeLabel(engine)}
                      {dataset && trainingKind(engine, dataset.targets.length)
                        ? ` · ${trainingKind(engine, dataset.targets.length)}`
                        : ""}
                    </dd>
                  )}
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Baseline</dt>
                  <dd className="mt-1">{baselineEngine?.name ?? "Choose a baseline"}</dd>
                </div>
              </dl>
              {engine && (
                <div className="space-y-2 border-t pt-4">
                  <p className="text-xs font-medium">Settings</p>
                  {changedSettings(engine.conditions, conditions, tasks, pinned).length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      Model defaults
                      {Object.keys(pinned).length
                        ? ", with settings fixed by pretrained weights"
                        : ""}
                    </p>
                  ) : (
                    <ul className="space-y-1 text-xs text-muted-foreground">
                      {changedSettings(engine.conditions, conditions, tasks, pinned).map((spec) => (
                        <li key={spec.key}>
                          {spec.label}:{" "}
                          <span className="font-medium text-foreground">
                            {settingValue(spec, conditions[spec.key] ?? spec.default)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {baselineEngine &&
                    changedSettings(
                      baselineEngine.conditions,
                      baselineConditions,
                      tasks,
                      baselinePinned,
                    ).length > 0 && (
                      <ul className="space-y-1 text-xs text-muted-foreground">
                        {changedSettings(
                          baselineEngine.conditions,
                          baselineConditions,
                          tasks,
                          baselinePinned,
                        ).map((spec) => (
                          <li key={spec.key}>
                            Baseline {spec.label}:{" "}
                            <span className="font-medium text-foreground">
                              {settingValue(spec, baselineConditions[spec.key] ?? spec.default)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  {Object.entries(pinned).map(([key, value]) => (
                    <p key={key} className="text-xs text-muted-foreground">
                      {engine.conditions.find((spec) => spec.key === key)?.label ?? key}:{" "}
                      {String(value)} · fixed by pretrained weights
                    </p>
                  ))}
                  {Object.entries(baselinePinned).map(([key, value]) => (
                    <p key={key} className="text-xs text-muted-foreground">
                      Baseline{" "}
                      {baselineEngine?.conditions.find((spec) => spec.key === key)?.label ?? key}:{" "}
                      {String(value)} · fixed by pretrained weights
                    </p>
                  ))}
                  {canTuneCutoffs && (
                    <p className="text-xs text-muted-foreground">
                      Decision cutoff: {tuneCutoffs ? "tuned on validation data" : "fixed at 0.5"}
                    </p>
                  )}
                </div>
              )}
              {engine && dataset && baselineEngine && (
                <div className="space-y-2 border-t pt-4">
                  <p className="text-xs font-medium">Planned work</p>
                  <ol className="space-y-2 text-xs text-muted-foreground">
                    <li className="flex gap-2">
                      <Check className="mt-0.5 size-3 shrink-0" />
                      <span>Train {engine.name} on the dataset split.</span>
                    </li>
                    {!selfCompare && (
                      <li className="flex gap-2">
                        <Check className="mt-0.5 size-3 shrink-0" />
                        <span>Train {baselineEngine.name} as the comparison baseline.</span>
                      </li>
                    )}
                    {hasRandomComparison(dataset.split.strategy) && (
                      <li className="flex gap-2">
                        <Check className="mt-0.5 size-3 shrink-0" />
                        <span>Train {engine.name} again on a random split for comparison.</span>
                      </li>
                    )}
                    <li className="flex gap-2">
                      <Check className="mt-0.5 size-3 shrink-0" />
                      <span>Generate a scorecard to review before publishing.</span>
                    </li>
                  </ol>
                  <div className="flex flex-wrap gap-1.5">
                    <Badge variant="outline" className="font-normal">
                      {hasRandomComparison(dataset.split.strategy)
                        ? selfCompare
                          ? 2
                          : 3
                        : selfCompare
                          ? 1
                          : 2}{" "}
                      training stages
                    </Badge>
                    {engine.lane === "gpu" && <Badge variant="secondary">GPU lane required</Badge>}
                  </div>
                </div>
              )}
              {issues.length > 0 && (
                <div
                  id="training-issues"
                  className="space-y-2 rounded-lg border bg-muted/30 p-3"
                  aria-live="polite"
                >
                  <p className="text-xs font-medium">Before you train</p>
                  <ul className="space-y-1">
                    {issues.map((issue) => (
                      <li key={issue.id}>
                        <button
                          type="button"
                          onClick={() => focusIssue(issue)}
                          className="text-left text-xs text-muted-foreground underline decoration-muted-foreground/40 underline-offset-2 hover:text-foreground"
                        >
                          {issue.message}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <Button
                className="w-full"
                onClick={submit}
                disabled={!canSubmit}
                aria-describedby={issues.length ? "training-issues" : undefined}
              >
                Train
              </Button>
              <p className="text-xs leading-relaxed text-muted-foreground">
                Training runs in the background. The resulting protocol stays a draft until you
                review its scorecard and publish it.
              </p>
              <Button variant="ghost" className="w-full" onClick={() => router.push("/protocols")}>
                Cancel
              </Button>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}
