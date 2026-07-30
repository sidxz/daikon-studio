"use client";

import { PREDICTION_TEMPLATE_CSV } from "@/features/datasets";
import { useProtocols } from "@/features/protocols";
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
import { showError } from "@/shared/lib/toast";
import { Download, FileUp } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import Papa from "papaparse";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useCreateRun, useUploadPredictionFile } from "../hooks/use-runs";

export function PredictWizard() {
  const router = useRouter();
  const params = useSearchParams();

  const [protocolId, setProtocolId] = useState(params.get("protocol") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [structureColumn, setStructureColumn] = useState("");

  const protocols = useProtocols();
  const upload = useUploadPredictionFile();
  const create = useCreateRun();

  // Only published protocols are runnable; a draft is not something anyone
  // else can rely on, so offering one here would just produce a 409.
  const published = (protocols.data?.items ?? []).filter((protocol) => protocol.status !== "draft");

  const onDrop = useCallback((files: File[]) => {
    const dropped = files[0];
    if (!dropped) return;
    Papa.parse<Record<string, string>>(dropped, {
      header: true,
      skipEmptyLines: true,
      preview: 5,
      complete: (result) => {
        const fields = (result.meta.fields ?? []).filter((field) => field.trim() !== "");
        if (fields.length === 0) {
          showError("No columns found. Is this a CSV with a header row?");
          return;
        }
        setFile(dropped);
        setColumns(fields);
        const guess =
          fields.find((field) => field.toLowerCase().trim() === "smiles") ??
          fields.find((field) => field.toLowerCase().includes("smiles")) ??
          fields[0];
        setStructureColumn(guess);
      },
      error: () => showError("Could not read that file"),
    });
  }, []);

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
      });
      router.push(`/runs/${run.id}`);
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
          Score your own compounds with a published protocol. You do not need to know anything about
          the model underneath — it carries its own units.
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
                Nothing is published yet. Train a protocol and publish it first.
              </p>
            )}
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
              <span className="text-xs text-muted-foreground">
                One column of SMILES. No measured values needed — that is what you are asking for.
              </span>
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
            <div className="space-y-1.5">
              <Label>Structure column</Label>
              <Select value={structureColumn} onValueChange={setStructureColumn}>
                <SelectTrigger>
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
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={() => router.push("/runs")} disabled={busy}>
          Cancel
        </Button>
        <Button onClick={submit} disabled={!protocolId || !file || busy}>
          {busy ? "Starting…" : "Run"}
        </Button>
      </div>
    </div>
  );
}
