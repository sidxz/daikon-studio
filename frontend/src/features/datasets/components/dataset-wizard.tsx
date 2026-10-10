"use client";

import { Explainer } from "@/shared/components/explainers/explainer";
import { SPLIT_MS, SplitFigure, splitCaption } from "@/shared/components/explainers/figures/split";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { Badge } from "@/shared/components/ui/badge";
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
import { Skeleton } from "@/shared/components/ui/skeleton";
import type { ApiError } from "@/shared/lib/api/custom-instance";
import { saveText } from "@/shared/lib/api/download";
import { SPLIT_VOCABULARY, isGroupedSplit } from "@/shared/lib/split";
import { showSuccess } from "@/shared/lib/toast";
import { Check, ChevronRight, Download, FileUp, Pencil } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  useDatasetPreview,
  useFreezeDatasetPreview,
  useStartDatasetPreview,
  useUploadDatasetFile,
} from "../hooks/use-datasets";
import {
  type ColumnRole,
  columnRole,
  draftFromUpload,
  replaceUpload,
  setColumnRole,
} from "../lib/draft-from-upload";
import { type CsvPreview, DATASET_TEMPLATE_CSV, parseCsvPreview } from "../lib/parse-csv";
import { ACCEPTED_UPLOADS, sizeLimitMb, toCsvFile } from "../lib/to-csv-file";
import {
  type DatasetDraft,
  type DraftTarget,
  EMPTY_DRAFT,
  SPLIT_COPY,
  type SplitStrategy,
  type ValidationReport,
} from "../types";
import { DatasetBuildProgress } from "./dataset-build-progress";
import { DatasetReadinessView } from "./dataset-readiness-view";
import { ValidationReportView } from "./validation-report-view";

const STEPS = ["Upload", "Columns & targets", "Split", "Review & create"] as const;
/**
 * Which kind of data each split is for. Scaffold is the recommendation for
 * small molecules, which is most datasets -- but a sequence dataset has no
 * Bemis–Murcko scaffold to split on, so naming the data rather than ranking
 * the strategies keeps "Recommended" from reading as "the others are worse".
 */
const SPLIT_BADGE: Record<SplitStrategy, string | null> = {
  scaffold: "Recommended for molecules",
  random: null,
  identity: "For protein sequences",
  position: "For variants of one protein",
  predefined: "For reproducing a published benchmark",
};
const ROLES: Record<ColumnRole, string> = {
  // Neutral on purpose: this role is chosen before the file is read, and the kind
  // is detected from the data, not declared here. Naming one modality told a
  // scientist uploading protein that their sequences were SMILES.
  structure: "Structure (SMILES or sequence)",
  identifier: "Identifier",
  target: "Target",
  unused: "Unused",
  split: "Split assignment (train/validation/test)",
};

export function DatasetWizard() {
  const router = useRouter();
  const params = useSearchParams();
  const resumed = params.get("review");
  const [step, setStep] = useState(resumed ? 3 : 0);
  const [furthest, setFurthest] = useState(resumed ? 3 : 0);
  const [draft, setDraft] = useState<DatasetDraft>(EMPTY_DRAFT);
  const [csv, setCsv] = useState<CsvPreview | null>(null);
  const [uploadRef, setUploadRef] = useState<string | null>(null);
  const [reviewId, setReviewId] = useState<string | null>(resumed);
  const [failure, setFailure] = useState<{
    message: string;
    report?: ValidationReport;
    duringCreation?: boolean;
  } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileReading, setFileReading] = useState(false);
  const [mappingNotice, setMappingNotice] = useState<string | null>(null);
  // The dropped workbook, kept so another of its sheets can be converted without
  // asking for the file again. Null for a CSV, which has nothing to choose.
  const [workbook, setWorkbook] = useState<{
    source: File;
    sheetNames: string[];
    sheet: string;
  } | null>(null);
  const upload = useUploadDatasetFile();
  const prepare = useStartDatasetPreview();
  const freeze = useFreezeDatasetPreview();
  const review = useDatasetPreview(reviewId);
  const preparation = review.data?.preparation;
  const working =
    fileReading ||
    upload.isPending ||
    prepare.isPending ||
    freeze.isPending ||
    review.data?.status === "running";

  function goTo(next: number) {
    setStep(next);
    setFurthest((previous) => Math.max(previous, next));
  }
  function invalidateReview() {
    setReviewId(null);
    setFailure(null);
    if (resumed) router.replace("/datasets/new", { scroll: false });
  }
  function patch(changes: Partial<DatasetDraft>) {
    if (Object.keys(changes).some((key) => key !== "name")) invalidateReview();
    setDraft((previous) => ({ ...previous, ...changes }));
  }
  function patchTarget(column: string, changes: Partial<DraftTarget>) {
    invalidateReview();
    setDraft((previous) => ({
      ...previous,
      targets: previous.targets.map((target) =>
        target.column === column ? { ...target, ...changes } : target,
      ),
    }));
  }

  // One file -- or one sheet of one workbook -- taken into the draft. The drop
  // handler and the sheet picker differ only in which sheet they ask for, so
  // the reading, previewing and re-mapping live here once.
  const adopt = useCallback(async (source: File, sheet?: string) => {
    const limit = sizeLimitMb(source.name);
    if (source.size > limit * 1024 * 1024) {
      throw new Error(`Choose a file smaller than ${limit} MB.`);
    }
    // Excel is converted at the door, so everything downstream -- the preview,
    // the column guesses and the upload -- sees the same CSV bytes the backend
    // will read.
    const converted = await toCsvFile(source, sheet);
    const parsed = await parseCsvPreview(converted.file);
    setCsv(parsed);
    setWorkbook(
      converted.sheet ? { source, sheetNames: converted.sheetNames, sheet: converted.sheet } : null,
    );
    setDraft((previous) =>
      previous.file || previous.structureColumn
        ? replaceUpload(previous, parsed.columns, parsed.rows, converted.file)
        : {
            ...draftFromUpload(parsed.columns, parsed.rows, converted.file.name),
            file: converted.file,
          },
    );
    setUploadRef(null);
    setReviewId(null);
    setFailure(null);
    return converted;
  }, []);

  const onDrop = useCallback(
    async (files: File[]) => {
      const file = files[0];
      if (!file) return;
      setFileReading(true);
      setFileError(null);
      try {
        const converted = await adopt(file);
        setMappingNotice(
          converted.sheetNames.length > 1
            ? `Read the sheet "${converted.sheet}" of ${converted.sheetNames.length} in this workbook. Check that it is the right sheet, and that the column roles and target types match your measurements.`
            : "Column roles and target types are suggestions. Check them against your measurements before continuing.",
        );
        router.replace("/datasets/new", { scroll: false });
        setStep(1);
        setFurthest(1);
      } catch (error) {
        setFileError(error instanceof Error ? error.message : "Could not read that file.");
      } finally {
        setFileReading(false);
      }
    },
    [adopt, router],
  );

  async function chooseSheet(sheet: string) {
    if (!workbook || sheet === workbook.sheet) return;
    setFileReading(true);
    setFileError(null);
    try {
      invalidateReview();
      await adopt(workbook.source, sheet);
      setMappingNotice(
        `Now reading the sheet "${sheet}". Check the column roles again — a different sheet can hold different columns.`,
      );
    } catch (error) {
      setFileError(error instanceof Error ? error.message : "Could not read that sheet.");
    } finally {
      setFileReading(false);
    }
  }

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: ACCEPTED_UPLOADS,
    multiple: false,
    noClick: true,
    disabled: working,
    onDropRejected: () => setFileError("Choose one CSV or Excel file: .csv, .xlsx, .xlsm or .xls."),
    maxSize: 100 * 1024 * 1024,
  });

  // A bookmarked review can be read without re-uploading or repeating preparation.
  useEffect(() => {
    if (!preparation || draft.structureColumn) return;
    setDraft({
      ...EMPTY_DRAFT,
      name: preparation.name,
      structureColumn: preparation.structure_column,
      idColumn: preparation.id_column,
      targets: preparation.targets.map((target) => ({
        column: target.column,
        kind: target.kind,
        unit: target.unit ?? "",
        direction: target.direction ?? "",
      })),
      strategy: preparation.split.strategy,
      seed: preparation.split.seed,
      splitColumn: preparation.split.column ?? null,
    });
  }, [preparation, draft.structureColumn]);

  useEffect(() => {
    if (review.data?.dataset_id) router.push(`/datasets/${review.data.dataset_id}`);
    if (review.data?.status !== "failed") return;
    const error = review.data.error;
    const detail = error?.detail;
    setFailure({
      message:
        typeof error?.message === "string" ? error.message : "Could not prepare this dataset.",
      report:
        detail && typeof detail === "object" && "total_rows" in detail
          ? (detail as ValidationReport)
          : undefined,
    });
  }, [review.data, router]);

  const mappingReason = !draft.name.trim()
    ? "Enter a dataset name."
    : draft.name.trim().length > 256
      ? "Use a name of 256 characters or fewer."
      : !draft.structureColumn
        ? "Choose a structure column."
        : !draft.targets.length
          ? "Choose at least one target column."
          : null;
  // The seed check is skipped for a predefined split rather than falling through to
  // it: the seed input is not rendered there (it decides nothing when the partitions
  // come from the file), so a NaN seed left over from another strategy would block the
  // step with a message and no field to act on.
  const splitReason =
    draft.strategy === "predefined"
      ? draft.splitColumn
        ? null
        : "Go back and mark the column that holds each row's partition."
      : !Number.isSafeInteger(draft.seed)
        ? "Enter a whole-number seed."
        : null;
  const reason =
    step === 0
      ? !draft.file
        ? "Upload a CSV to continue."
        : null
      : step === 1
        ? mappingReason
        : step === 2
          ? (splitReason ??
            mappingReason ??
            (!draft.file ? "Upload a file to prepare a new review." : null))
          : !preparation
            ? "Prepare a successful review before creating the dataset."
            : !draft.name.trim()
              ? "Enter a dataset name."
              : null;

  async function prepareReview() {
    if (!draft.file || mappingReason || splitReason) return;
    setFailure(null);
    goTo(3);
    try {
      const ref = uploadRef ?? (await upload.mutateAsync(draft.file));
      setUploadRef(ref);
      const started = await prepare.mutateAsync({
        name: draft.name.trim(),
        upload_ref: ref,
        file_name: draft.file.name,
        structure_column: draft.structureColumn,
        id_column: draft.idColumn,
        deduplicate: draft.deduplicate,
        targets: draft.targets.map((target) => ({
          column: target.column,
          kind: target.kind,
          unit: target.kind === "numeric" && target.unit.trim() ? target.unit.trim() : null,
          direction: target.kind === "numeric" && target.direction ? target.direction : null,
        })),
        split: {
          strategy: draft.strategy,
          seed: draft.seed,
          ...(draft.strategy === "predefined" ? { column: draft.splitColumn } : {}),
        },
      });
      setReviewId(started.id);
      router.replace(`/datasets/new?review=${started.id}`, { scroll: false });
    } catch (error) {
      const apiError = error as ApiError;
      if (!apiError?.silent)
        setFailure({
          message: error instanceof Error ? error.message : "Could not prepare the dataset.",
        });
    }
  }

  async function create() {
    if (!reviewId || !preparation) return;
    try {
      const dataset = await freeze.mutateAsync({ id: reviewId, name: draft.name.trim() });
      showSuccess("Dataset created");
      router.push(`/datasets/${dataset.id}`);
    } catch (error) {
      const apiError = error as ApiError;
      if (!apiError?.silent)
        setFailure({
          message: error instanceof Error ? error.message : "Could not create the dataset.",
          duringCreation: true,
        });
    }
  }

  function edit(next: number) {
    if ((next === 1 || next === 2) && !csv) {
      setMappingNotice(
        "Upload the original or a replacement file to edit its preparation. Your saved column choices will be kept where possible.",
      );
      goTo(0);
    } else goTo(next);
  }

  return (
    <div className="w-full min-w-0 space-y-6">
      <PageHeader
        title="New dataset"
        description="Define what your data means, check how it will be prepared, then create a reproducible dataset."
      />
      <div className="max-w-5xl space-y-5">
        <ol aria-label="Dataset setup progress" className="flex flex-wrap gap-2">
          {STEPS.map((label, index) => (
            <li key={label} className="flex items-center gap-2">
              <Button
                variant={index === step ? "secondary" : "ghost"}
                size="sm"
                aria-current={index === step ? "step" : undefined}
                disabled={working || index > furthest}
                onClick={() => edit(index)}
              >
                <span className="flex size-5 items-center justify-center rounded-full border text-xs">
                  {index < step ? <Check className="size-3" /> : index + 1}
                </span>
                {label}
              </Button>
              {index < STEPS.length - 1 && (
                <ChevronRight className="size-3 text-muted-foreground" aria-hidden="true" />
              )}
            </li>
          ))}
        </ol>

        {(draft.file || preparation?.file_name) && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-muted/20 px-4 py-3">
            <div className="flex items-center gap-3">
              <FileUp className="size-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">
                  {workbook?.source.name ?? draft.file?.name ?? preparation?.file_name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {draft.file
                    ? `${(draft.file.size / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} KB · first rows shown below`
                    : "Previously uploaded file · prepared for this review"}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {/* A supplementary workbook usually holds several tables, and the
                  one that matters is rarely the first sheet. */}
              {workbook && workbook.sheetNames.length > 1 && (
                <Select value={workbook.sheet} onValueChange={chooseSheet} disabled={working}>
                  <SelectTrigger
                    aria-label="Sheet to import"
                    className="h-8 w-auto min-w-44 bg-background text-xs"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {workbook.sheetNames.map((name) => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              <Button variant="outline" size="sm" disabled={working} onClick={open}>
                Replace file
              </Button>
            </div>
          </div>
        )}

        {fileError && (
          <p role="alert" className="text-sm text-destructive">
            {fileError}
          </p>
        )}

        {/* Keep the file input mounted so replacement works at every step. */}
        <input {...getInputProps()} />
        {step === 0 && (
          <Card>
            <CardContent className="space-y-4 py-6">
              <div {...getRootProps()}>
                <button
                  type="button"
                  disabled={working}
                  onClick={open}
                  className={`flex min-h-64 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors ${isDragActive ? "border-primary bg-primary/5" : "border-border hover:bg-muted/40"}`}
                >
                  <FileUp className="size-9 text-muted-foreground" />
                  <span className="font-medium">
                    {fileReading
                      ? "Reading the preview…"
                      : "Drop a CSV or Excel file here, or choose one"}
                  </span>
                  <span className="max-w-sm text-sm text-muted-foreground">
                    A structure column — SMILES or an amino-acid sequence — and at least one
                    measured target. CSV up to 100 MB, Excel (.xlsx, .xlsm, .xls) up to 25 MB.
                  </span>
                </button>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground">
                  Other columns can be kept as identifiers or left unused.
                </p>
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
              {mappingNotice && (
                <p aria-live="polite" className="text-sm text-muted-foreground">
                  {mappingNotice}
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {step === 1 && csv && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Define columns and targets</CardTitle>
              <p className="text-sm text-muted-foreground">
                Assign each column a role, then describe the targets you want to predict.
              </p>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="max-w-lg space-y-1.5">
                <Label htmlFor="dataset-name">Dataset name</Label>
                <Input
                  id="dataset-name"
                  value={draft.name}
                  maxLength={256}
                  onChange={(event) => patch({ name: event.target.value })}
                />
              </div>
              <p aria-live="polite" className="text-xs text-muted-foreground">
                {mappingNotice ??
                  "Suggestions are based on column names and the first 20 rows. Full validation happens before creation."}
              </p>
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-xs">
                  <caption className="sr-only">Column roles and sampled values</caption>
                  <thead>
                    <tr>
                      {csv.columns.map((column, index) => (
                        <th
                          key={column}
                          className="min-w-48 border-b bg-muted/30 px-3 py-3 text-left align-top"
                        >
                          <Label
                            htmlFor={`role-${index}`}
                            className="mb-2 block truncate font-mono"
                          >
                            {column}
                          </Label>
                          <Select
                            value={columnRole(draft, column)}
                            onValueChange={(value) => {
                              invalidateReview();
                              setDraft((previous) =>
                                setColumnRole(previous, column, value as ColumnRole, csv.rows),
                              );
                              setMappingNotice(
                                `Updated ${column} to ${ROLES[value as ColumnRole].toLowerCase()}. Each column has one role.`,
                              );
                            }}
                          >
                            <SelectTrigger
                              id={`role-${index}`}
                              aria-label={`Role for ${column}`}
                              className="bg-background"
                            >
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {Object.entries(ROLES).map(([role, label]) => (
                                <SelectItem key={role} value={role}>
                                  {label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csv.rows.slice(0, 4).map((row, index) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: sampled file rows never reorder
                      <tr key={`preview-${index}`} className="border-b last:border-0">
                        {csv.columns.map((column) => (
                          <td
                            key={column}
                            className="max-w-56 truncate px-3 py-2 font-mono text-muted-foreground"
                            title={row[column]}
                          >
                            {row[column] || <span className="italic">Empty</span>}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="space-y-4">
                {draft.targets.map((target, index) => (
                  <div key={target.column} className="rounded-lg border p-4">
                    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm font-medium">
                        Predict <span className="font-mono">{target.column}</span>
                      </p>
                      <Badge variant="outline" className="font-normal">
                        {target.kind === "binary" ? "0 / 1 labels" : "Continuous values"}
                      </Badge>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-3">
                      <div className="space-y-1.5">
                        <Label htmlFor={`kind-${index}`}>Target type</Label>
                        <Select
                          value={target.kind}
                          onValueChange={(kind) =>
                            patchTarget(target.column, { kind: kind as DraftTarget["kind"] })
                          }
                        >
                          <SelectTrigger id={`kind-${index}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="numeric">Measured value</SelectItem>
                            <SelectItem value="binary">Active or inactive</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                          {target.kind === "binary"
                            ? "Full validation requires 0 (inactive) or 1 (active)."
                            : "Examples: pIC50, solubility, or permeability."}
                        </p>
                      </div>
                      {target.kind === "numeric" && (
                        <>
                          <div className="space-y-1.5">
                            <Label htmlFor={`unit-${index}`}>Unit (optional)</Label>
                            <Input
                              id={`unit-${index}`}
                              value={target.unit}
                              onChange={(event) =>
                                patchTarget(target.column, { unit: event.target.value })
                              }
                              placeholder="µM, log mol/L…"
                            />
                            <p className="text-xs text-muted-foreground">
                              A display label; values are not converted.
                            </p>
                          </div>
                          <div className="space-y-1.5">
                            <Label htmlFor={`direction-${index}`}>Preferred direction</Label>
                            <Select
                              value={target.direction || "unspecified"}
                              onValueChange={(value) =>
                                patchTarget(target.column, {
                                  direction:
                                    value === "unspecified" ? "" : (value as "high" | "low"),
                                })
                              }
                            >
                              <SelectTrigger id={`direction-${index}`}>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="unspecified">No preference</SelectItem>
                                <SelectItem value="high">Higher is better</SelectItem>
                                <SelectItem value="low">Lower is better</SelectItem>
                              </SelectContent>
                            </Select>
                            <p className="text-xs text-muted-foreground">
                              Check this choice. It guides ranking of predictions.
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {step === 2 && (
          <Card>
            <CardHeader>
              <CardTitle id="split-strategy-label" className="text-base">
                Choose how to evaluate generalization
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                The split separates learning, tuning, and final evaluation.
              </p>
            </CardHeader>
            <CardContent className="space-y-5">
              <div aria-labelledby="split-strategy-label" className="grid gap-3 sm:grid-cols-2">
                {(Object.keys(SPLIT_COPY) as SplitStrategy[]).map((strategy) => (
                  <label
                    key={strategy}
                    className={`cursor-pointer rounded-xl border p-4 ${draft.strategy === strategy ? "border-primary bg-primary/5" : "border-border"}`}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="split-strategy"
                        value={strategy}
                        checked={draft.strategy === strategy}
                        onChange={() => patch({ strategy })}
                        className="accent-primary"
                      />
                      <span className="text-sm font-medium">{SPLIT_COPY[strategy].title}</span>
                      {SPLIT_BADGE[strategy] && (
                        <Badge variant="secondary">{SPLIT_BADGE[strategy]}</Badge>
                      )}
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      {SPLIT_COPY[strategy].detail}
                    </p>
                  </label>
                ))}
              </div>
              {draft.strategy === "predefined" ? (
                <div className="rounded-lg bg-muted/30 p-4">
                  <p className="text-sm font-medium">
                    Partitions come from{" "}
                    <span className="font-mono">{draft.splitColumn ?? "a column you choose"}</span>
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Rows are assigned exactly as your file says. The review shows the exact counts.
                    A validation partition is optional: many published benchmarks have only training
                    and test rows.
                  </p>
                </div>
              ) : (
                <div className="rounded-lg bg-muted/30 p-4">
                  <p className="text-sm font-medium">
                    Intended split: 80% training · 10% validation · 10% test
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {isGroupedSplit(draft.strategy)
                      ? `Each ${SPLIT_VOCABULARY[draft.strategy].group} stays on one side, so actual counts can differ. The review shows the exact result.`
                      : "The review shows the exact result."}
                  </p>
                </div>
              )}
              <details className="rounded-lg border p-4">
                <summary className="cursor-pointer text-sm font-medium">
                  Why this split matters
                </summary>
                <div className="mt-4">
                  <Explainer
                    id="split"
                    durationMs={SPLIT_MS}
                    caption={splitCaption(draft.strategy)}
                    replayKey={draft.strategy}
                  >
                    {(t) => <SplitFigure t={t} strategy={draft.strategy} />}
                  </Explainer>
                </div>
              </details>
              {draft.strategy !== "predefined" && (
                <details className="rounded-lg border p-4">
                  <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
                  <div className="mt-4 max-w-xs space-y-1.5">
                    <Label htmlFor="seed">Split seed</Label>
                    <Input
                      id="seed"
                      type="number"
                      step={1}
                      value={Number.isNaN(draft.seed) ? "" : draft.seed}
                      aria-invalid={Boolean(splitReason)}
                      onChange={(event) =>
                        patch({
                          seed: event.target.value === "" ? Number.NaN : Number(event.target.value),
                        })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Saved with the dataset so the split is reproducible.
                    </p>
                    {splitReason && <p className="text-xs text-destructive">{splitReason}</p>}
                  </div>
                </details>
              )}
              {draft.strategy === "predefined" && splitReason && (
                <p className="text-xs text-destructive">{splitReason}</p>
              )}
              <details className="rounded-lg border p-4">
                <summary className="cursor-pointer text-sm font-medium">
                  Repeated measurements
                </summary>
                <div className="mt-4 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="deduplicate"
                      checked={draft.deduplicate}
                      onCheckedChange={(value) => patch({ deduplicate: value !== false })}
                    />
                    <Label htmlFor="deduplicate" className="font-normal">
                      Combine repeated measurements of the same compound
                    </Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    On by default. Switching it off keeps your file's rows exactly as they are,
                    which is what reproducing a published row count needs. The cost is that one
                    compound can then appear in both training and test, and a compound measured as
                    both active and inactive will no longer be caught.
                  </p>
                </div>
              </details>
            </CardContent>
          </Card>
        )}

        {step === 3 && (
          <div className="space-y-5">
            {review.data?.status === "running" ? (
              <DatasetBuildProgress build={review.data} reviewing />
            ) : upload.isPending || prepare.isPending || review.isLoading ? (
              <Card>
                <CardContent className="space-y-3 py-6">
                  <p className="text-sm font-medium">Preparing your review…</p>
                  <Skeleton className="h-3 w-full" />
                  <p className="text-xs text-muted-foreground">No dataset has been created yet.</p>
                </CardContent>
              </Card>
            ) : null}
            {review.isError && (
              <QueryError
                title="Could not load this review"
                description={review.error.message}
                retry={() => review.refetch()}
                retrying={review.isFetching}
              />
            )}
            {failure && (
              <div
                role="alert"
                className="space-y-4 rounded-lg border border-destructive/40 bg-destructive/5 p-4"
              >
                <p className="text-sm font-medium text-destructive">{failure.message}</p>
                <p className="text-xs text-muted-foreground">
                  {failure.duringCreation
                    ? "Creation could not be confirmed. You can retry safely or check the dataset list."
                    : "No dataset was created. Edit the columns or split, replace the file, or try preparation again."}
                </p>
                {failure.report && <ValidationReportView report={failure.report} rejected />}
                {draft.file && (
                  <Button variant="outline" disabled={working} onClick={prepareReview}>
                    Prepare review again
                  </Button>
                )}
              </div>
            )}
            {preparation && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Review your dataset</CardTitle>
                    <p className="text-sm text-muted-foreground">
                      This is the exact preparation that will be saved. Review exclusions and target
                      definitions before creating it.
                    </p>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="space-y-1.5">
                      <Label htmlFor="review-name">Dataset name</Label>
                      <Input
                        id="review-name"
                        value={draft.name}
                        maxLength={256}
                        onChange={(event) =>
                          setDraft((previous) => ({ ...previous, name: event.target.value }))
                        }
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">Columns and targets</p>
                      <Button variant="ghost" size="sm" onClick={() => edit(1)}>
                        <Pencil className="size-3" />
                        Edit columns
                      </Button>
                    </div>
                    <dl className="grid gap-3 text-sm sm:grid-cols-2">
                      <div>
                        <dt className="text-xs text-muted-foreground">Structure column</dt>
                        <dd className="mt-1 font-mono">{preparation.structure_column}</dd>
                      </div>
                      <div>
                        <dt className="text-xs text-muted-foreground">Identifier</dt>
                        <dd className="mt-1 font-mono">{preparation.id_column ?? "None"}</dd>
                      </div>
                    </dl>
                    <ul className="space-y-2">
                      {preparation.targets.map((target) => (
                        <li
                          key={target.column}
                          className="flex flex-wrap gap-2 rounded-md bg-muted/30 p-3 text-sm"
                        >
                          <span className="font-mono font-medium">{target.column}</span>
                          <span className="text-muted-foreground">
                            {target.kind === "binary"
                              ? "Active / inactive (0 / 1)"
                              : "Measured value"}
                            {target.unit
                              ? ` · ${target.unit}`
                              : target.kind === "numeric"
                                ? " · unit unspecified"
                                : ""}
                            {target.kind === "numeric"
                              ? ` · ${target.direction === "high" ? "higher is better" : target.direction === "low" ? "lower is better" : "no preferred direction"}`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
                    <CardTitle className="text-base">
                      Actual split · {SPLIT_COPY[preparation.split.strategy].title}
                    </CardTitle>
                    <Button variant="ghost" size="sm" onClick={() => edit(2)}>
                      <Pencil className="size-3" />
                      Edit split
                    </Button>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-xs text-muted-foreground">
                      {preparation.split.column
                        ? `Partitions from "${preparation.split.column}"`
                        : `Seed ${preparation.split.seed}`}{" "}
                      · {preparation.readiness.row_count.toLocaleString()}{" "}
                      {draft.deduplicate ? "unique compounds" : "rows"}
                    </p>
                    <DatasetReadinessView readiness={preparation.readiness} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Preparation results</CardTitle>
                    <p className="text-sm text-muted-foreground">
                      {draft.deduplicate
                        ? "Invalid rows and conflicting labels are excluded. Repeated numeric measurements are averaged per compound; repeated agreeing binary labels are collapsed."
                        : "Invalid rows are excluded. Repeated measurements of the same compound are kept as separate rows, so the same compound can appear in both training and test."}
                    </p>
                  </CardHeader>
                  <CardContent>
                    <ValidationReportView report={preparation.validation_report} />
                  </CardContent>
                </Card>
                <p className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
                  Creating this dataset fixes its structures, targets, and split for reproducible
                  training. To change those later, create a new dataset. This review is available
                  for 24 hours.
                </p>
              </>
            )}
          </div>
        )}

        <div className="sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-background/95 p-4 backdrop-blur">
          <Button
            variant="ghost"
            disabled={working}
            onClick={() => (step === 0 ? router.push("/datasets") : edit(step - 1))}
          >
            {step === 0 ? "Cancel" : "Back"}
          </Button>
          <div className="flex flex-wrap items-center justify-end gap-3">
            {reason && !working && (
              <p
                id="dataset-next-reason"
                className="max-w-sm text-xs text-muted-foreground"
                aria-live="polite"
              >
                {reason}
              </p>
            )}
            <Button
              disabled={Boolean(reason) || working}
              aria-describedby={reason ? "dataset-next-reason" : undefined}
              onClick={() =>
                step < 2
                  ? goTo(step + 1)
                  : step === 2
                    ? preparation
                      ? goTo(3)
                      : prepareReview()
                    : create()
              }
            >
              {working
                ? freeze.isPending
                  ? "Creating…"
                  : "Preparing…"
                : step === 3
                  ? "Create dataset"
                  : step === 2
                    ? preparation
                      ? "Review dataset"
                      : "Prepare review"
                    : "Continue"}
              <ChevronRight className="size-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
