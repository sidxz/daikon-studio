"use client";

import { PREDICTION_TEMPLATE_CSV, useDataset } from "@/features/datasets";
import { useEngines } from "@/features/engines";
import { useProtocols } from "@/features/protocols";
import type { Protocol } from "@/features/protocols";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent } from "@/shared/components/ui/card";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { saveText } from "@/shared/lib/api/download";
import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { showError } from "@/shared/lib/toast";
import { Download, FileUp } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import Papa from "papaparse";
import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useCreateRun, useUploadPredictionFile } from "../hooks/use-runs";
import { summarisePreview } from "../lib/parse-preview";
import { PredictionPreview } from "./prediction-preview";

// Radix forbids an empty item value, and a blank header is dropped on parse, so
// no real column can ever be named this.
const NO_ID_COLUMN = " ";

function ProtocolContext({ protocol }: { protocol: Protocol }) {
  const { data: dataset } = useDataset(protocol.dataset_id);
  const { data: engines } = useEngines();
  const engine = engines?.find((candidate) => candidate.id === protocol.engine_id);
  const conditions = engine?.conditions ?? [];

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3 text-sm">
      <p>
        {/* A Protocol declares one readout per numeric target and two per binary
            one (probability and class), so the list can run to many. */}
        Predicts{" "}
        <span className="font-medium">
          {new Intl.ListFormat("en-US", { type: "conjunction" }).format(
            protocol.readouts.map((readout) =>
              readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
            ),
          )}
        </span>
        {dataset && (
          <span className="text-muted-foreground">
            {" "}
            · trained on {dataset.row_count} compounds from {dataset.name}
          </span>
        )}
      </p>
      {conditions.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Trained with
          </p>
          {/* Read-only, and not an oversight: neither engine reads conditions
              at predict time, so an input here would be a control that changes
              nothing. When an engine declares predict-time conditions, this
              becomes ConditionFields. */}
          <dl className="mt-1 flex flex-wrap gap-x-6 gap-y-1 text-xs">
            {conditions.map((condition) => (
              <div key={condition.key} className="flex gap-1.5">
                <dt className="text-muted-foreground">{condition.label}</dt>
                <dd className="font-mono">
                  {String(protocol.conditions?.[condition.key] ?? condition.default ?? "—")}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}

export function PredictWizard() {
  const router = useRouter();
  const params = useSearchParams();

  const [protocolId, setProtocolId] = useState(params.get("protocol") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [structureColumn, setStructureColumn] = useState("");
  const [idColumn, setIdColumn] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, string | undefined>[]>([]);

  const protocols = useProtocols(undefined, 200);
  const upload = useUploadPredictionFile();
  const create = useCreateRun();

  // Only published protocols are runnable; a draft is not something anyone
  // else can rely on, so offering one here would just produce a 409.
  const published = (protocols.data?.items ?? []).filter((protocol) => protocol.status !== "draft");
  const selectedProtocol = published.find((protocol) => protocol.id === protocolId);

  const onDrop = useCallback((files: File[]) => {
    const dropped = files[0];
    if (!dropped) return;
    // ponytail: parses the whole file on the main thread, synchronously,
    // before the preview can show a true row count. Bounded today by the
    // shared upload endpoint's 100 MB cap (`MAX_UPLOAD_BYTES`, datasets.py) --
    // at that ceiling a pathological CSV blocks the thread between drop and
    // preview. Upgrade path: papaparse's `worker: true`, once that stall is
    // actually felt.
    Papa.parse<Record<string, string>>(dropped, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const fields = (result.meta.fields ?? []).filter((field) => field.trim() !== "");
        if (fields.length === 0) {
          showError("No columns found. The file must be a CSV with a header row.");
          return;
        }
        setFile(dropped);
        setColumns(fields);
        setRows(result.data);
        const guess =
          fields.find((field) => field.toLowerCase().trim() === "smiles") ??
          fields.find((field) => field.toLowerCase().includes("smiles")) ??
          fields[0];
        setStructureColumn(guess);
        setIdColumn(guessIdColumn(fields, guess));
      },
      error: () => showError("Could not read the file"),
    });
  }, []);

  const summary = useMemo(
    () => (rows.length > 0 && structureColumn ? summarisePreview(rows, structureColumn) : null),
    [rows, structureColumn],
  );
  const compoundCount = summary ? summary.total - summary.blank : 0;

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
    noClick: true,
  });

  async function submit() {
    if (!file) return;
    try {
      const uploadRef = await upload.mutateAsync(file);
      const run = await create.mutateAsync({
        protocol_id: protocolId,
        upload_ref: uploadRef,
        structure_column: structureColumn,
        id_column: idColumn,
      });
      // A cache hit comes back 202 with an already-ready Run, so the status is
      // the only way to tell that no work was started. Saying so beats showing
      // a progress bar that was never going to move.
      const cached = run.status === "ready" ? "&cached=1" : "";
      router.push(`/runs/${run.id}?compounds=${compoundCount}${cached}`);
    } catch {
      // The global mutation handler already surfaced the message.
    }
  }

  const busy = upload.isPending || create.isPending;

  return (
    <div className="mx-auto w-full max-w-2xl space-y-4 p-2">
      <div>
        <h1 className="text-lg font-semibold">Run a protocol</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Predict properties for your compounds with a published protocol. Results are reported in
          the protocol's units.
        </p>
      </div>

      <Card>
        <CardContent className="min-h-[20rem] space-y-4 py-6">
          <div className="space-y-1.5">
            <Label>Protocol</Label>
            <Select value={protocolId} onValueChange={setProtocolId}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a published protocol" />
              </SelectTrigger>
              <SelectContent>
                {published.map((protocol) => (
                  <SelectItem key={protocol.id} value={protocol.id}>
                    {protocol.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {protocols.data && published.length === 0 && (
              <p className="text-xs text-muted-foreground">
                No published protocols. Train and publish a protocol first.
              </p>
            )}
            {selectedProtocol && <ProtocolContext protocol={selectedProtocol} />}
          </div>

          <div {...getRootProps()} className="space-y-2">
            <input {...getInputProps()} />
            <Label>Compounds</Label>
            <button
              type="button"
              onClick={open}
              className={`flex w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
                isDragActive ? "border-primary bg-primary/5" : "border-border hover:bg-muted/40"
              }`}
            >
              <FileUp className="size-6 text-muted-foreground" />
              <span className="text-sm font-medium">
                {file ? file.name : "Drop a CSV of structures, or click to choose one"}
              </span>
              <span className="text-xs text-muted-foreground">Requires one SMILES column.</span>
            </button>
            <div className="flex items-center justify-end">
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  saveText(PREDICTION_TEMPLATE_CSV, "daikon-studio-prediction-template.csv")
                }
              >
                <Download className="size-4" />
                Download template
              </Button>
            </div>
          </div>

          {columns.length > 1 && (
            <>
              <div className="space-y-1.5">
                <Label htmlFor="structure-column">Structure column</Label>
                <Select value={structureColumn} onValueChange={setStructureColumn}>
                  <SelectTrigger id="structure-column">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {columns.map((column) => (
                      <SelectItem key={column} value={column}>
                        {column}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="id-column">Identifier column (optional)</Label>
                <Select
                  value={idColumn ?? NO_ID_COLUMN}
                  onValueChange={(value) => setIdColumn(value === NO_ID_COLUMN ? null : value)}
                >
                  <SelectTrigger id="id-column">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={NO_ID_COLUMN}>None</SelectItem>
                    {columns.map((column) => (
                      <SelectItem key={column} value={column}>
                        {column}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Included with each prediction and in the export, so results can be matched to your
                  file.
                </p>
              </div>
            </>
          )}
          {summary && <PredictionPreview summary={summary} column={structureColumn} />}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={() => router.push("/runs")} disabled={busy}>
          Cancel
        </Button>
        <Button onClick={submit} disabled={!protocolId || !file || compoundCount === 0 || busy}>
          {busy
            ? "Starting…"
            : `Predict ${compoundCount} compound${compoundCount === 1 ? "" : "s"}`}
        </Button>
      </div>
    </div>
  );
}
