"use client";

import {
  ACCEPTED_UPLOADS,
  PREDICTION_TEMPLATE_CSV,
  sizeLimitMb,
  toCsvFile,
  useDataset,
} from "@/features/datasets";
import { useEngines } from "@/features/engines";
import { useProtocolOptions } from "@/features/protocols";
import type { Protocol } from "@/features/protocols";
import { PageHeader } from "@/shared/components/page-header";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { saveText } from "@/shared/lib/api/download";
import type { ChemCellarImportResponse } from "@/shared/lib/api/model";
import { useAppConfig } from "@/shared/lib/app-config";
import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { showError } from "@/shared/lib/toast";
import { Download, FileUp } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import Papa from "papaparse";
import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useCreateRun, useUploadPredictionFile } from "../hooks/use-runs";
import { activeCompounds, formatRunDate } from "../lib/chemcellar-runs";
import { summarisePreview } from "../lib/parse-preview";
import { ChemCellarPicker } from "./chemcellar-picker";
import { PredictionPreview } from "./prediction-preview";
import { ProtocolPicker } from "./protocol-picker";

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
            protocol.readouts
              .filter((readout) => readout.type !== "probability")
              .map((readout) =>
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
        <details className="mt-3">
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            Training settings
          </summary>
          {/* Read-only, and not an oversight: neither engine reads conditions
              at predict time, so an input here would be a control that changes
              nothing. When an engine declares predict-time conditions, this
              becomes ConditionFields. */}
          <dl className="mt-1 flex flex-wrap gap-x-6 gap-y-1 text-xs">
            {conditions.map((condition) => (
              <div key={condition.key} className="flex gap-1.5">
                <dt className="text-muted-foreground">{condition.label}</dt>
                <dd className="font-mono">
                  {String(protocol.conditions?.[condition.key] ?? condition.default ?? "N/A")}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </div>
  );
}

export function PredictWizard() {
  const router = useRouter();
  const params = useSearchParams();
  const { chemcellarUrl } = useAppConfig();

  const [protocolId, setProtocolId] = useState(params.get("protocol") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [structureColumn, setStructureColumn] = useState("");
  const [idColumn, setIdColumn] = useState<string | null>(null);
  const [tab, setTab] = useState<"csv" | "chemcellar">("csv");
  const [imported, setImported] = useState<ChemCellarImportResponse | null>(null);
  const [name, setName] = useState("");
  const [rows, setRows] = useState<Record<string, string | undefined>[]>([]);

  const protocols = useProtocolOptions();
  const upload = useUploadPredictionFile();
  const create = useCreateRun();

  // Only published protocols are runnable; a draft is not something anyone
  // else can rely on, so offering one here would just produce a 409.
  const published = (protocols.data ?? []).filter((protocol) => protocol.status !== "draft");
  const selectedProtocol = published.find((protocol) => protocol.id === protocolId);

  const onDrop = useCallback(async (files: File[]) => {
    const original = files[0];
    if (!original) return;
    // Excel becomes CSV at the door, so the parse below and the upload that
    // follows both see the same bytes the backend will read.
    let dropped: File;
    try {
      if (original.size > sizeLimitMb(original.name) * 1024 * 1024) {
        throw new Error(`Choose a file smaller than ${sizeLimitMb(original.name)} MB.`);
      }
      const converted = await toCsvFile(original);
      dropped = converted.file;
      if (converted.sheetNames.length > 1) {
        // No sheet picker on this screen, unlike the dataset wizard: a
        // prediction input is a list of structures, not a supplementary
        // workbook. Naming the sheet is enough for someone to notice the
        // wrong one and export the right one themselves.
        showError(
          `Read the sheet "${converted.sheet}" of ${converted.sheetNames.length}. Export a single sheet if that is not the one you meant.`,
        );
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : "Could not read that file.");
      return;
    }
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
        setName(dropped.name.replace(/\.[^.]*$/, "").slice(0, 200));
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
    accept: ACCEPTED_UPLOADS,
    multiple: false,
    noClick: true,
  });

  const { count, ready } = activeCompounds(tab, compoundCount, imported);

  async function submit() {
    try {
      let run: Awaited<ReturnType<typeof create.mutateAsync>>;
      if (tab === "chemcellar") {
        if (!imported) return;
        run = await create.mutateAsync({
          protocol_id: protocolId,
          upload_ref: imported.upload_ref,
          structure_column: "smiles",
          id_column: "compound_id",
          name: name.trim() || undefined,
        });
      } else {
        if (!file) return;
        const uploadRef = await upload.mutateAsync(file);
        run = await create.mutateAsync({
          protocol_id: protocolId,
          upload_ref: uploadRef,
          structure_column: structureColumn,
          id_column: idColumn,
          name: name.trim() || undefined,
        });
      }
      // A cache hit comes back 202 with an already-ready Run, so the status is
      // the only way to tell that no work was started. Saying so beats showing
      // a progress bar that was never going to move.
      const cached = run.status === "ready" ? "&cached=1" : "";
      router.push(`/runs/${run.id}?compounds=${count}${cached}`);
    } catch {
      // The global mutation handler already surfaced the message.
    }
  }

  const busy = upload.isPending || create.isPending;

  const cellarPane = (
    <>
      <ChemCellarPicker
        onImported={(value) => {
          setImported(value);
          if (value)
            setName(`${value.source.protocol_name} · ${value.source.run_date}`.slice(0, 200));
        }}
      />
      {imported && (
        <>
          <PredictionPreview
            summary={{ total: imported.compound_count, blank: 0, sample: imported.sample }}
            column="smiles"
          />
          <p className="text-sm text-muted-foreground">
            From ChemCellar: {imported.source.protocol_name}, run of{" "}
            {formatRunDate(imported.source.run_date)}
          </p>
          {imported.without_structure > 0 && (
            <p className="text-xs text-muted-foreground">
              {imported.without_structure} compound
              {imported.without_structure === 1 ? "" : "s"} in this run{" "}
              {imported.without_structure === 1 ? "has" : "have"} no disclosed structure and{" "}
              {imported.without_structure === 1 ? "is" : "are"} not included.
            </p>
          )}
        </>
      )}
    </>
  );

  const csvPane = (
    <>
      <div {...getRootProps()} className="space-y-2">
        <input {...getInputProps()} />
        {!chemcellarUrl && <Label>Compounds</Label>}
        <button
          type="button"
          onClick={open}
          className={`flex w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
            isDragActive ? "border-primary bg-primary/5" : "border-border hover:bg-muted/40"
          }`}
        >
          <FileUp className="size-6 text-muted-foreground" />
          <span className="text-sm font-medium">
            {file ? file.name : "Drop a CSV or Excel file of structures, or click to choose one"}
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
    </>
  );

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Run a protocol"
        description="Predict properties for your compounds with a published protocol. Results are reported in the protocol’s units."
      />

      <div className="w-full min-w-0 max-w-3xl space-y-4">
        <Card>
          <CardContent className="min-h-[20rem] space-y-4 py-6">
            <div className="space-y-1.5">
              <Label htmlFor="prediction-protocol">Protocol</Label>
              <ProtocolPicker
                id="prediction-protocol"
                protocols={published}
                value={protocolId}
                onChange={setProtocolId}
                loading={protocols.isLoading}
              />
              {protocols.isError && (
                <p role="alert" className="text-sm text-destructive">
                  Could not load protocols.{" "}
                  <button type="button" className="underline" onClick={() => protocols.refetch()}>
                    Try again
                  </button>
                </p>
              )}
              {protocols.data && published.length === 0 && (
                <div className="rounded-lg border border-dashed bg-muted/20 p-4">
                  <p className="text-sm font-medium">No published protocols</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Train a protocol and review its scorecard, or publish an existing draft.
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button asChild size="sm">
                      <Link href="/protocols/new">Train a protocol</Link>
                    </Button>
                    <Button asChild size="sm" variant="outline">
                      <Link href="/protocols">Review protocols</Link>
                    </Button>
                  </div>
                </div>
              )}
              {selectedProtocol && <ProtocolContext protocol={selectedProtocol} />}
            </div>

            {chemcellarUrl ? (
              <div className="space-y-2">
                <Label>Compounds</Label>
                <Tabs value={tab} onValueChange={(value) => setTab(value as "csv" | "chemcellar")}>
                  <TabsList>
                    <TabsTrigger value="csv">Upload CSV</TabsTrigger>
                    <TabsTrigger value="chemcellar">From ChemCellar</TabsTrigger>
                  </TabsList>
                  {/* Both panes stay mounted, so the picker keeps its choice across
                    tab switches and always agrees with what Predict submits. */}
                  <TabsContent
                    value="csv"
                    forceMount
                    className="space-y-4 data-[state=inactive]:hidden"
                  >
                    {csvPane}
                  </TabsContent>
                  <TabsContent
                    value="chemcellar"
                    forceMount
                    className="space-y-4 data-[state=inactive]:hidden"
                  >
                    {cellarPane}
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              csvPane
            )}

            <div className="space-y-1.5">
              <Label htmlFor="run-name">Run name</Label>
              <Input
                id="run-name"
                value={name}
                maxLength={200}
                onChange={(e) => setName(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Shown in the runs list. Defaults to the file name.
              </p>
            </div>
          </CardContent>
        </Card>

        <div className="flex items-center justify-between">
          <Button variant="ghost" onClick={() => router.push("/runs")} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={!protocolId || !ready || busy}>
            {busy ? "Starting…" : `Predict ${count} compound${count === 1 ? "" : "s"}`}
          </Button>
        </div>
        {selectedProtocol && (
          <p className="text-right text-xs text-muted-foreground">
            Using {selectedProtocol.name}
            {selectedProtocol.protocol_version != null &&
              ` · v${selectedProtocol.protocol_version}`}
          </p>
        )}
      </div>
    </div>
  );
}
