"use client";

import { PageHeader } from "@/shared/components/page-header";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import { Download } from "lucide-react";
import Link from "next/link";
import { useCollection, useExportCollection } from "../hooks/use-collections";

export function CollectionDetail({ collectionId }: { collectionId: string }) {
  const { data: collection, isLoading, isError } = useCollection(collectionId);
  const exporter = useExportCollection();

  useBreadcrumbTrail(
    collection
      ? [{ label: "Collections", href: "/collections" }, { label: collection.name }]
      : null,
  );

  if (isLoading) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError || !collection) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load this collection</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title={collection.name}
        description={
          <>
            {collection.member_count.toLocaleString()} compounds · saved{" "}
            {new Date(collection.created_at).toLocaleString()}
          </>
        }
        action={
          <>
            <Button
              variant="outline"
              disabled={exporter.isPending}
              onClick={() =>
                exporter.mutate({ id: collection.id, name: collection.name, format: "csv" })
              }
            >
              <Download className="size-4" />
              Export CSV
            </Button>
            <Button
              variant="outline"
              disabled={exporter.isPending}
              onClick={() =>
                exporter.mutate({ id: collection.id, name: collection.name, format: "sdf" })
              }
            >
              <Download className="size-4" />
              Export SDF
            </Button>
          </>
        }
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Provenance</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="font-normal">
              {collection.provenance.generation_method === "ai_predicted"
                ? "AI-predicted"
                : collection.provenance.generation_method}
            </Badge>
            <span className="text-sm text-muted-foreground">
              These values are model predictions.
            </span>
          </div>
          <p className="text-sm">
            <Link
              href={`/runs/${collection.derived_from_run_id}`}
              className="underline underline-offset-2"
            >
              The run these were triaged from
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
