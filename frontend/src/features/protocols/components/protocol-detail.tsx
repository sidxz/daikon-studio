"use client";

import { useDataset } from "@/features/datasets";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/shared/components/ui/alert-dialog";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { ApiError } from "@/shared/lib/api/custom-instance";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
import { useState } from "react";
import { useProtocol, usePublishProtocol, useScorecard } from "../hooks/use-protocols";
import { ProtocolRuns } from "./protocol-runs";
import { ScorecardView } from "./scorecard-view";

export function ProtocolDetail({ protocolId }: { protocolId: string }) {
  const { data: protocol, isLoading, isError } = useProtocol(protocolId);
  const scorecard = useScorecard(protocolId);
  const { data: dataset } = useDataset(protocol?.dataset_id);
  const publish = usePublishProtocol();
  const [confirming, setConfirming] = useState(false);

  useBreadcrumbTrail(
    protocol ? [{ label: "Protocols", href: "/protocols" }, { label: protocol.name }] : null,
  );

  if (isLoading) {
    return (
      <div className="mx-auto w-full max-w-5xl space-y-4 p-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError || !protocol) {
    return (
      <div className="mx-auto w-full max-w-5xl p-2">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load this protocol</p>
        </div>
      </div>
    );
  }

  const published = protocol.status !== "draft";

  return (
    <div className="mx-auto w-full max-w-5xl space-y-4 p-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">{protocol.name}</h1>
            <Badge variant={published ? "default" : "outline"} className="font-normal">
              {published ? "Published" : "Draft"}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {protocol.engine_id}
            {dataset && (
              <>
                {" · trained on "}
                <Link href={`/datasets/${dataset.id}`} className="underline underline-offset-2">
                  {dataset.name}
                </Link>
              </>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          {published ? (
            <Button asChild>
              <Link href={`/runs/new?protocol=${protocol.id}`}>Run this protocol</Link>
            </Button>
          ) : (
            <Button onClick={() => setConfirming(true)} disabled={publish.isPending}>
              {publish.isPending ? "Publishing…" : "Publish"}
            </Button>
          )}
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            What it predicts
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            {protocol.readouts.map((readout) => (
              <span key={readout.name}>
                <span className="font-mono">{readout.name}</span>
                {readout.unit && <span className="ml-1 text-muted-foreground">{readout.unit}</span>}
                {readout.direction && (
                  <span className="ml-1 text-xs text-muted-foreground">
                    ({readout.direction} is better)
                  </span>
                )}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      {scorecard.isLoading && <Skeleton className="h-64 w-full" />}
      {scorecard.data && <ScorecardView scorecard={scorecard.data} />}

      <ProtocolRuns protocolId={protocol.id} />

      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Publish {protocol.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              Publishing locks this protocol permanently. Its weights, dataset, split and metrics
              can never change again, which is what makes it citable — and it means this cannot be
              undone. Anyone in this workspace will be able to run it.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={publish.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={publish.isPending}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                publish.mutate(protocolId, {
                  onSuccess: () => setConfirming(false),
                  onError: (error) => {
                    // 423 means someone already published it -- not a failure
                    // worth alarming about, just a stale view.
                    if (error instanceof ApiError && error.status === 423) {
                      setConfirming(false);
                    }
                  },
                });
              }}
            >
              {publish.isPending ? "Publishing…" : "Publish permanently"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
