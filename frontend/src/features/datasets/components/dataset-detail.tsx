"use client";

import { NotebookPanel } from "@/features/pages";
import { PageHeader } from "@/shared/components/page-header";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { isComputing, useDataset, useDatasetProfile } from "../hooks/use-datasets";
import { SPLIT_COPY } from "../types";
import { CompoundBrowser } from "./compound-browser";
import { DatasetProfileSkeleton, DatasetProfileView } from "./dataset-profile-view";
import { DeleteDatasetButton } from "./delete-dataset-button";
import { IdColumnField } from "./id-column-field";
import { ProfileComputing } from "./profile-computing";
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
  const tab = useSearchParams().get("tab");
  const { data: dataset, isLoading, isError, error } = useDataset(datasetId);
  const creator = useMemberName()(dataset?.created_by);
  // Fetched alongside the Dataset rather than on tab activation: the first ever
  // request starts the server computing the profile (minutes for a large
  // dataset), so it is better started while the reader is still on the overview.
  const [profileTarget, setProfileTarget] = useState(0);
  const profile = useDatasetProfile(datasetId, profileTarget);

  // Declared explicitly so the breadcrumb never prints the id from the URL.
  useBreadcrumbTrail(
    dataset ? [{ label: "Datasets", href: "/datasets" }, { label: dataset.name }] : null,
  );

  if (isLoading) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError || !dataset) {
    return (
      <div className="w-full min-w-0 space-y-4">
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
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title={dataset.name}
        description={
          <>
            {dataset.row_count.toLocaleString()} compounds · frozen{" "}
            {new Date(dataset.created_at).toLocaleString()}
            {creator && ` · Created by ${creator}`}
          </>
        }
        action={
          <>
            {dataset.can_delete && <DeleteDatasetButton dataset={dataset} />}
            <Button asChild>
              <Link href={`/protocols/new?dataset=${dataset.id}`}>Train a protocol</Link>
            </Button>
          </>
        }
      />

      <Tabs defaultValue={tab === "notebook" ? "notebook" : "overview"}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="diversity">Diversity</TabsTrigger>
          <TabsTrigger value="compounds">Compounds</TabsTrigger>
          <TabsTrigger value="notebook">Notebook</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What this predicts</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid gap-4 sm:grid-cols-3">
                <Field
                  label="Structures"
                  value={<span className="font-mono">{dataset.structure_column}</span>}
                />
                <Field
                  label={dataset.targets.length === 1 ? "Target" : "Targets"}
                  value={
                    <ul className="space-y-1">
                      {dataset.targets.map((target) => (
                        <li key={target.column}>
                          <span className="font-mono">{target.column}</span>
                          <span className="ml-2 text-muted-foreground">
                            {target.kind === "numeric" ? "Measured value" : "Active / inactive"}
                            {target.unit ? ` · ${target.unit}` : ""}
                            {target.direction
                              ? ` · ${target.direction === "high" ? "higher" : "lower"} is better`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  }
                />
                <Field label="Identifier column" value={<IdColumnField dataset={dataset} />} />
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
                <Field
                  label="Seed"
                  value={<span className="font-mono">{dataset.split.seed}</span>}
                />
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
            <h2 className="mb-2 text-sm font-medium">Validation report</h2>
            <ValidationReportView report={dataset.validation_report} />
          </div>
        </TabsContent>

        <TabsContent value="diversity" className="space-y-4">
          {dataset.targets.length > 1 && (
            <div className="flex items-center gap-2">
              <Label htmlFor="profile-target" className="text-sm font-normal">
                Profile for
              </Label>
              <Select
                value={String(profileTarget)}
                onValueChange={(value) => setProfileTarget(Number(value))}
              >
                <SelectTrigger id="profile-target" className="w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {dataset.targets.map((target, index) => (
                    <SelectItem key={target.column} value={String(index)}>
                      {target.column}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {profile.isLoading ? (
            <DatasetProfileSkeleton />
          ) : profile.isError || !profile.data ? (
            <div className="rounded-lg border border-border p-4">
              <p className="text-sm font-medium">Could not load the dataset profile</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {profile.error instanceof Error
                  ? profile.error.message
                  : "Its frozen snapshot could not be read."}
              </p>
            </div>
          ) : isComputing(profile.data) ? (
            <ProfileComputing
              structureKind={dataset?.validation_report?.structure_kind}
              startedAt={profile.data.started_at}
              compounds={profile.data.compounds}
            />
          ) : (
            <DatasetProfileView
              dataset={dataset}
              target={dataset.targets[profileTarget]}
              profile={profile.data}
            />
          )}
        </TabsContent>

        <TabsContent value="compounds">
          <CompoundBrowser dataset={dataset} />
        </TabsContent>

        <TabsContent value="notebook">
          <NotebookPanel kind="dataset" ownerId={dataset.id} canCreate={dataset.can_edit} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
