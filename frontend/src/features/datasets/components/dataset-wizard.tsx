"use client";

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
import type { ApiError } from "@/shared/lib/api/custom-instance";
import { saveText } from "@/shared/lib/api/download";
import { showError } from "@/shared/lib/toast";
import { Download, FileUp } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useCreateDataset, useUploadDatasetFile } from "../hooks/use-datasets";
import {
  type CsvPreview,
  DATASET_TEMPLATE_CSV,
  guessStructureColumn,
  looksBinary,
  parseCsvPreview,
} from "../lib/parse-csv";
import {
  type DatasetDraft,
  EMPTY_DRAFT,
  SPLIT_COPY,
  TARGET_KIND_COPY,
  type ValidationReport,
} from "../types";
import { ValidationReportView } from "./validation-report-view";

const STEPS = ["File", "Columns", "Target", "Split"] as const;

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

export function DatasetWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState<DatasetDraft>(EMPTY_DRAFT);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [rejection, setRejection] = useState<ValidationReport | null>(null);

  const upload = useUploadDatasetFile();
  const create = useCreateDataset();
  const patch = (changes: Partial<DatasetDraft>) => setDraft((prev) => ({ ...prev, ...changes }));

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    try {
      const parsed = await parseCsvPreview(file);
      const structureColumn = guessStructureColumn(parsed.columns);
      const remaining = parsed.columns.filter((column) => column !== structureColumn);
      const targetColumn = remaining[0] ?? "";
      setPreview(parsed);
      setDraft({
        ...EMPTY_DRAFT,
        file,
        name: file.name.replace(/\.csv$/i, ""),
        structureColumn,
        targetColumn,
        kind: targetColumn && looksBinary(parsed.rows, targetColumn) ? "binary" : "numeric",
      });
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
      const dataset = await create.mutateAsync({
        name: draft.name.trim(),
        upload_ref: uploadRef,
        structure_column: draft.structureColumn,
        target: {
          column: draft.targetColumn,
          kind: draft.kind,
          unit: draft.kind === "numeric" && draft.unit.trim() ? draft.unit.trim() : null,
          direction: draft.kind === "numeric" && draft.direction ? draft.direction : null,
        },
        split: { strategy: draft.strategy, seed: draft.seed },
      });
      router.push(`/datasets/${dataset.id}`);
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

  const busy = upload.isPending || create.isPending;
  const canContinue = [
    Boolean(draft.file),
    Boolean(draft.name.trim() && draft.structureColumn && draft.targetColumn),
    Boolean(draft.targetColumn),
    Boolean(draft.strategy),
  ][step];

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
                  A SMILES column and a target column are required. Other columns are not used for
                  training.
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
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label>Structures</Label>
                  <Select
                    value={draft.structureColumn}
                    onValueChange={(value) => patch({ structureColumn: value })}
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
                  <Label>Value to predict</Label>
                  <Select
                    value={draft.targetColumn}
                    onValueChange={(value) =>
                      patch({
                        targetColumn: value,
                        kind: looksBinary(preview.rows, value) ? "binary" : "numeric",
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {preview.columns
                        .filter((column) => column !== draft.structureColumn)
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
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>
                  What kind of value is <span className="font-mono">{draft.targetColumn}</span>?
                </Label>
                <div className="grid gap-2 sm:grid-cols-2">
                  {(["numeric", "binary"] as const).map((kind) => (
                    <button
                      key={kind}
                      type="button"
                      onClick={() => patch({ kind })}
                      className={`h-full rounded-lg border p-3 text-left transition-colors ${
                        draft.kind === kind
                          ? "border-primary bg-primary/5"
                          : "border-border hover:bg-muted/40"
                      }`}
                    >
                      <span className="text-sm font-medium">{TARGET_KIND_COPY[kind].title}</span>
                      <span className="mt-1 block text-xs text-muted-foreground">
                        {TARGET_KIND_COPY[kind].detail}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {draft.kind === "numeric" && (
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label htmlFor="unit">Unit</Label>
                    <Input
                      id="unit"
                      value={draft.unit}
                      onChange={(event) => patch({ unit: event.target.value })}
                      placeholder="µM, log mol/L, kcal/mol…"
                    />
                    <p className="text-xs text-muted-foreground">
                      Shown with every predicted value.
                    </p>
                  </div>
                  <div className="space-y-1.5">
                    <Label>Preferred direction</Label>
                    <Select
                      value={draft.direction || "high"}
                      onValueChange={(value) => patch({ direction: value as "high" | "low" })}
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
