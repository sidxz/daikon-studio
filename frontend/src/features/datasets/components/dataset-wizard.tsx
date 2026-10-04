"use client";

import { Explainer } from "@/shared/components/explainers/explainer";
import { SPLIT_MS, SplitFigure, splitCaption } from "@/shared/components/explainers/figures/split";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Checkbox } from "@/shared/components/ui/checkbox";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import type { ApiError } from "@/shared/lib/api/custom-instance";
import { saveText } from "@/shared/lib/api/download";
import { showError, showSuccess } from "@/shared/lib/toast";
import { Download, FileUp } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useDatasetBuild, useStartDatasetBuild, useUploadDatasetFile } from "../hooks/use-datasets";
import { draftFromUpload, toggleTarget, withColumns } from "../lib/draft-from-upload";
import {
  type CsvPreview,
  DATASET_TEMPLATE_CSV,
  guessStructureColumn,
  parseCsvPreview,
} from "../lib/parse-csv";
import {
  type DatasetDraft,
  type DraftTarget,
  EMPTY_DRAFT,
  SPLIT_COPY,
  TARGET_KIND_COPY,
  type ValidationReport,
} from "../types";
import { DatasetBuildProgress } from "./dataset-build-progress";
import { ValidationReportView } from "./validation-report-view";

const STEPS = ["File", "Columns", "Targets", "Split"] as const;

function StepIndicator({ current }: { current: number }) {
  return (
    <ol className="flex items-center gap-2 text-xs">
      {STEPS.map((label, index) => {
        const state = index === current ? "current" : index < current ? "done" : "todo";
        return (
          <li key={label} className="flex items-center gap-2">
            <span
              className={
                state === "current"
                  ? "rounded-full bg-primary px-2.5 py-0.5 font-medium text-primary-foreground"
                  : state === "done"
                    ? "rounded-full bg-muted px-2.5 py-0.5 font-medium"
                    : "rounded-full px-2.5 py-0.5 text-muted-foreground"
              }
            >
              {label}
            </span>
            {index < STEPS.length - 1 && <span className="text-muted-foreground">·</span>}
          </li>
        );
      })}
    </ol>
  );
}

/** The Select value for "no identifier column"; Radix needs a non-empty value. */
const NO_ID = "__none__";
export function DatasetWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState<DatasetDraft>(EMPTY_DRAFT);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [rejection, setRejection] = useState<ValidationReport | null>(null);

  const upload = useUploadDatasetFile();
  const start = useStartDatasetBuild();
  const [buildId, setBuildId] = useState<string | null>(null);
  const build = useDatasetBuild(buildId);
  const patch = (changes: Partial<DatasetDraft>) => setDraft((prev) => ({ ...prev, ...changes }));
  function patchTarget(column: string, changes: Partial<DraftTarget>) {
    setDraft((prev) => ({
      ...prev,
      targets: prev.targets.map((target) =>
        target.column === column ? { ...target, ...changes } : target,
      ),
    }));
  }

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    try {
      const parsed = await parseCsvPreview(file);
      setPreview(parsed);
      setDraft({ ...draftFromUpload(parsed.columns, parsed.rows, file.name), file });
      setStep(1);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Could not read that file");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
    noClick: true,
  });

  async function submit() {
    if (!draft.file) return;
    setRejection(null);
    try {
      const uploadRef = await upload.mutateAsync(draft.file);
      const started = await start.mutateAsync({
        name: draft.name.trim(),
        upload_ref: uploadRef,
        structure_column: draft.structureColumn,
        id_column: draft.idColumn,
        targets: draft.targets.map((target) => ({
          column: target.column,
          kind: target.kind,
          unit: target.kind === "numeric" && target.unit.trim() ? target.unit.trim() : null,
          direction: target.kind === "numeric" && target.direction ? target.direction : null,
        })),
        split: { strategy: draft.strategy, seed: draft.seed },
      });
      setBuildId(started.id);
    } catch (error) {
      const apiError = error as ApiError;
      // A 401 the session renewal is already handling: nothing to say here.
      if (apiError?.silent) return;
      // A 422 body IS the report. Render it; never reduce it to a toast.
      const body = apiError?.body as { detail?: unknown } | undefined;
      const detail = body?.detail;
      if (
        apiError?.status === 422 &&
        detail &&
        typeof detail === "object" &&
        "total_rows" in detail
      ) {
        setRejection(detail as ValidationReport);
        return;
      }
      showError(apiError instanceof Error ? apiError.message : "Could not create the dataset");
    }
  }

  // A build that has ended: open the dataset, or show why it was refused -- a
  // rejected file's validation report, exactly as a rejected request shows it.
  useEffect(() => {
    if (build.isError) {
      setBuildId(null);
      showError("Lost contact with the server during the build. Check the dataset list.");
      return;
    }
    const ended = build.data;
    if (!ended || ended.status === "running") return;
    setBuildId(null);
    if (ended.status === "succeeded" && ended.dataset_id) {
      showSuccess("Dataset frozen");
      router.push(`/datasets/${ended.dataset_id}`);
      return;
    }
    const detail = ended.error?.detail;
    if (detail && typeof detail === "object" && "total_rows" in detail) {
      setRejection(detail as ValidationReport);
      return;
    }
    const message = ended.error?.message;
    showError(typeof message === "string" ? message : "Could not create the dataset");
  }, [build.data, build.isError, router]);

  const busy = upload.isPending || start.isPending || buildId !== null;
  const canContinue = [
    Boolean(draft.file),
    Boolean(draft.name.trim() && draft.structureColumn && draft.targets.length > 0),
    draft.targets.length > 0,
    Boolean(draft.strategy),
  ][step];

  if (buildId !== null && build.data?.status === "running") {
    return (
      <div className="p-2">
        <DatasetBuildProgress build={build.data} />
      </div>
    );
  }

  if (rejection) {
    return (
      <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
        <div>
          <h1 className="text-lg font-semibold text-destructive">Validation failed</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            No dataset was created. Correct the rows listed below and upload the file again.
          </p>
        </div>
        <ValidationReportView report={rejection} rejected />
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => router.push("/datasets")}>
            Cancel
          </Button>
          <Button onClick={() => setRejection(null)}>Back to the wizard</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl space-y-4 p-2">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-lg font-semibold">New dataset</h1>
        <StepIndicator current={step} />
      </div>

      {/* Fixed min-height so the panel never resizes between steps. */}
      <Card>
        <CardContent className="min-h-[22rem] py-6">
          {step === 0 && (
            <div {...getRootProps()} className="flex h-full flex-col">
              <input {...getInputProps()} />
              <button
                type="button"
                onClick={open}
                className={`flex flex-1 cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                  isDragActive ? "border-primary bg-primary/5" : "border-border hover:bg-muted/40"
                }`}
              >
                <FileUp className="size-8 text-muted-foreground" />
                <span className="text-sm font-medium">Drop a CSV here, or click to choose one</span>
                <span className="max-w-sm text-xs text-muted-foreground">
                  A SMILES column and at least one target column are required. Other columns are not
                  used for training.
                </span>
              </button>
              <div className="mt-4 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Not sure of the format?</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    saveText(DATASET_TEMPLATE_CSV, "daikon-studio-dataset-template.csv")
                  }
                >
                  <Download className="size-4" />
                  Download template
                </Button>
              </div>
            </div>
          )}

          {step === 1 && preview && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="dataset-name">Name</Label>
                <Input
                  id="dataset-name"
                  value={draft.name}
                  onChange={(event) => patch({ name: event.target.value })}
                  placeholder="e.g. ESOL aqueous solubility"
                />
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-1.5">
                  <Label>Structures</Label>
                  <Select
                    value={draft.structureColumn}
                    onValueChange={(value) =>
                      setDraft((prev) => withColumns(prev, { structureColumn: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {preview.columns.map((column) => (
                        <SelectItem key={column} value={column}>
                          {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Target columns</Label>
                  <div className="max-h-48 space-y-1.5 overflow-y-auto rounded-md border p-2">
                    {preview.columns.map((column, index) =>
                      column === draft.structureColumn ? null : (
                        <div key={column} className="flex items-center gap-2">
                          <Checkbox
                            id={`target-${index}`}
                            checked={draft.targets.some((target) => target.column === column)}
                            onCheckedChange={(checked) =>
                              setDraft((prev) =>
                                toggleTarget(prev, column, checked === true, preview.rows),
                              )
                            }
                          />
                          <Label htmlFor={`target-${index}`} className="font-mono font-normal">
                            {column}
                          </Label>
                        </div>
                      ),
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Choose one or more. Every compound needs a value in each.
                  </p>
                </div>
                <div className="space-y-1.5">
                  <Label>Identifier (optional)</Label>
                  <Select
                    value={draft.idColumn ?? NO_ID}
                    onValueChange={(value) => patch({ idColumn: value === NO_ID ? null : value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NO_ID}>None</SelectItem>
                      {preview.columns
                        .filter(
                          (column) =>
                            column !== draft.structureColumn &&
                            !draft.targets.some((target) => target.column === column),
                        )
                        .map((column) => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b bg-muted/40 text-left">
                      {preview.columns.map((column) => (
                        <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.slice(0, 4).map((row, index) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: preview rows have no id and never reorder
                      <tr key={index} className="border-b last:border-0">
                        {preview.columns.map((column) => (
                          <td
                            key={column}
                            className="max-w-56 truncate px-3 py-1.5 font-mono text-muted-foreground"
                          >
                            {row[column]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-6">
              {draft.targets.map((target) => (
                <div key={target.column} className="space-y-4">
                  <div className="space-y-2">
                    <Label>
                      What kind of value is <span className="font-mono">{target.column}</span>?
                    </Label>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {(["numeric", "binary"] as const).map((kind) => (
                        <button
                          key={kind}
                          type="button"
                          onClick={() => patchTarget(target.column, { kind })}
                          className={`h-full rounded-lg border p-3 text-left transition-colors ${
                            target.kind === kind
                              ? "border-primary bg-primary/5"
                              : "border-border hover:bg-muted/40"
                          }`}
                        >
                          <span className="text-sm font-medium">
                            {TARGET_KIND_COPY[kind].title}
                          </span>
                          <span className="mt-1 block text-xs text-muted-foreground">
                            {TARGET_KIND_COPY[kind].detail}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                  {target.kind === "numeric" && (
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <Label htmlFor={`unit-${target.column}`}>Unit</Label>
                        <Input
                          id={`unit-${target.column}`}
                          value={target.unit}
                          onChange={(event) =>
                            patchTarget(target.column, { unit: event.target.value })
                          }
                          placeholder="µM, log mol/L, kcal/mol…"
                        />
                        <p className="text-xs text-muted-foreground">
                          Shown with every predicted value.
                        </p>
                      </div>
                      <div className="space-y-1.5">
                        <Label>Preferred direction</Label>
                        <Select
                          value={target.direction || "high"}
                          onValueChange={(value) =>
                            patchTarget(target.column, { direction: value as "high" | "low" })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="high">Higher is better</SelectItem>
                            <SelectItem value="low">Lower is better</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                          Used when ranking triage results and when judging a model against its
                          baseline.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>How should the test set be held out?</Label>
                <div className="grid gap-2">
                  {(["scaffold", "random"] as const).map((strategy) => (
                    <button
                      key={strategy}
                      type="button"
                      onClick={() => patch({ strategy })}
                      className={`h-full rounded-lg border p-3 text-left transition-colors ${
                        draft.strategy === strategy
                          ? "border-primary bg-primary/5"
                          : "border-border hover:bg-muted/40"
                      }`}
                    >
                      <span className="text-sm font-medium">
                        {SPLIT_COPY[strategy].title}
                        {strategy === "scaffold" && (
                          <span className="ml-2 text-xs font-normal text-muted-foreground">
                            Recommended
                          </span>
                        )}
                      </span>
                      <span className="mt-1 block text-xs text-muted-foreground">
                        {SPLIT_COPY[strategy].detail}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <Explainer
                id="split"
                durationMs={SPLIT_MS}
                caption={splitCaption(draft.strategy)}
                replayKey={draft.strategy}
              >
                {(t) => <SplitFigure t={t} strategy={draft.strategy} />}
              </Explainer>
              <div className="w-40 space-y-1.5">
                <Label htmlFor="seed">Seed</Label>
                <Input
                  id="seed"
                  type="number"
                  value={draft.seed}
                  onChange={(event) => patch({ seed: Number(event.target.value) })}
                />
                <p className="text-xs text-muted-foreground">
                  Frozen with the dataset, so the split is reproducible and citable.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          onClick={() => (step === 0 ? router.push("/datasets") : setStep(step - 1))}
          disabled={busy}
        >
          {step === 0 ? "Cancel" : "Back"}
        </Button>
        {step < STEPS.length - 1 ? (
          <Button onClick={() => setStep(step + 1)} disabled={!canContinue || busy}>
            Continue
          </Button>
        ) : (
          <Button onClick={submit} disabled={!canContinue || busy}>
            {busy ? "Validating…" : "Validate and freeze"}
          </Button>
        )}
      </div>
    </div>
  );
}
