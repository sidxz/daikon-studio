"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
import { useDataset } from "../hooks/use-datasets";
import { SPLIT_COPY } from "../types";
import { ValidationReportView } from "./validation-report-view";

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-sm">{value}</dd>
    </div>
  );
}

export function DatasetDetail({ datasetId }: { datasetId: string }) {
  const { data: dataset, isLoading, isError, error } = useDataset(datasetId);

  // Declared explicitly so the breadcrumb never prints the id from the URL.
  useBreadcrumbTrail(
    dataset ? [{ label: "Datasets", href: "/datasets" }, { label: dataset.name }] : null,
  );

  if (isLoading) {
    return (
      <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError || !dataset) {
    return (
      <div className="mx-auto w-full max-w-4xl p-2">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load this dataset</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {error instanceof Error ? error.message : "It may not exist in this workspace."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">{dataset.name}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {dataset.row_count.toLocaleString()} compounds · frozen{" "}
            {new Date(dataset.created_at).toLocaleString()}
          </p>
        </div>
        <Button asChild>
          <Link href={`/protocols/new?dataset=${dataset.id}`}>Train a protocol</Link>
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">What this predicts</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-4 sm:grid-cols-4">
            <Field
              label="Structures"
              value={<span className="font-mono">{dataset.structure_column}</span>}
            />
            <Field
              label="Target"
              value={<span className="font-mono">{dataset.target.column}</span>}
            />
            <Field
              label="Kind"
              value={dataset.target.kind === "numeric" ? "Measured value" : "Active / inactive"}
            />
            <Field
              label="Unit and direction"
              value={
                dataset.target.unit || dataset.target.direction ? (
                  <span className="font-mono">
                    {dataset.target.unit ?? "—"}
                    {dataset.target.direction ? ` · ${dataset.target.direction} is better` : ""}
                  </span>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )
              }
            />
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            Split
            <Badge variant="outline" className="font-normal">
              {SPLIT_COPY[dataset.split.strategy].title}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-sm text-muted-foreground">
            {SPLIT_COPY[dataset.split.strategy].detail}
          </p>
          <dl className="grid gap-4 sm:grid-cols-3">
            <Field label="Seed" value={<span className="font-mono">{dataset.split.seed}</span>} />
            <Field
              label="Train / validation / test"
              value={
                <span className="font-mono">
                  {(dataset.split.fractions ?? [0.8, 0.1, 0.1]).join(" / ")}
                </span>
              }
            />
            <Field
              label="Content hash"
              value={
                <span className="font-mono text-xs">{dataset.content_hash.slice(0, 16)}…</span>
              }
            />
          </dl>
        </CardContent>
      </Card>

      <div>
        <h2 className="mb-2 text-sm font-medium">What the file contained</h2>
        <ValidationReportView report={dataset.validation_report} />
      </div>
    </div>
  );
}
