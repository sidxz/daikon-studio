"use client";

import { useDataset } from "@/features/datasets";
import { PageHeader } from "@/shared/components/page-header";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { ApiError } from "@/shared/lib/api/custom-instance";
import type { ScorecardResponse } from "@/shared/lib/api/model";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
import { useState } from "react";
import { useProtocol, usePublishProtocol, useScorecard } from "../hooks/use-protocols";
import { formatCutoff } from "../lib/format-cutoff";
import { DeleteProtocolButton } from "./delete-protocol-button";
import { ProtocolChemicalSpace } from "./protocol-chemical-space";
import { ProtocolRuns } from "./protocol-runs";
import { ScorecardCompounds } from "./scorecard-compounds";
import { ScorecardView } from "./scorecard-view";

/**
 * One scorecard per target. One target renders exactly as before; several share a
 * header that says whether one model learned them all or each has its own, so a
 * row of per-target numbers is never mistaken for joint learning.
 */
export function Scorecards({
  scorecards,
  protocolId,
  view = "performance",
  selectedTarget,
  onTargetChange,
}: {
  scorecards: ScorecardResponse[];
  protocolId?: string;
  view?: "performance" | "compounds";
  selectedTarget?: string;
  onTargetChange?: (target: string) => void;
}) {
  const [first] = scorecards;
  if (!first)
    return <p className="text-sm text-muted-foreground">No test results are available yet.</p>;
  const TargetView = view === "compounds" ? ScorecardCompounds : ScorecardView;
  if (scorecards.length === 1)
    return <TargetView key={first.target} scorecard={first} protocolId={protocolId} />;
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        {first.joint_model
          ? `One model learned all ${scorecards.length} targets jointly. Each tab scores it on one target.`
          : `${scorecards.length} separate models, one per target, trained on the same compounds and split.`}
      </p>
      <Tabs
        defaultValue={first.target}
        value={
          selectedTarget == null
            ? undefined
            : scorecards.some((card) => card.target === selectedTarget)
              ? selectedTarget
              : first.target
        }
        onValueChange={onTargetChange}
      >
        <TabsList>
          {scorecards.map((card) => (
            <TabsTrigger key={card.target} value={card.target} className="font-mono">
              {card.target}
            </TabsTrigger>
          ))}
        </TabsList>
        {scorecards.map((card) => (
          <TabsContent key={card.target} value={card.target}>
            <TargetView scorecard={card} protocolId={protocolId} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}

export function ProtocolDetail({ protocolId }: { protocolId: string }) {
  const { data: protocol, isLoading, isError, error } = useProtocol(protocolId);
  const creator = useMemberName()(protocol?.created_by);
  const scorecard = useScorecard(protocolId);
  const { data: dataset } = useDataset(protocol?.dataset_id);
  const publish = usePublishProtocol();
  const [confirming, setConfirming] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<string>();

  useBreadcrumbTrail(
    protocol ? [{ label: "Protocols", href: "/protocols" }, { label: protocol.name }] : null,
  );

  if (isLoading) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError || !protocol) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <div className="w-full min-w-0 space-y-4">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">
            {missing
              ? "This protocol does not exist in this workspace."
              : "Could not load this protocol"}
          </p>
        </div>
      </div>
    );
  }

  const published = protocol.status !== "draft";

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title={
          <span className="inline-flex flex-wrap items-center gap-2">
            {protocol.name}
            <Badge variant={published ? "default" : "outline"} className="font-normal">
              {published ? "Published" : "Draft"}
            </Badge>
          </span>
        }
        description={
          <>
            {protocol.engine_id}
            {dataset && (
              <>
                {" · trained on "}
                <Link href={`/datasets/${dataset.id}`} className="underline underline-offset-2">
                  {dataset.name}
                </Link>
              </>
            )}
            {creator && ` · Created by ${creator}`}
          </>
        }
        action={
          <>
            {protocol.can_delete && <DeleteProtocolButton protocol={protocol} />}
            {published ? (
              <Button asChild>
                <Link href={`/runs/new?protocol=${protocol.id}`}>Run this protocol</Link>
              </Button>
            ) : (
              <Button onClick={() => setConfirming(true)} disabled={publish.isPending}>
                {publish.isPending ? "Publishing…" : "Publish"}
              </Button>
            )}
          </>
        }
      />

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Predicted readouts
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
                    ({readout.direction === "low" ? "lower" : "higher"} is better)
                  </span>
                )}
                {readout.type === "class" && readout.threshold != null && (
                  <span className="ml-1 text-xs text-muted-foreground">
                    class at cutoff {formatCutoff(readout.threshold)}
                  </span>
                )}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="performance" className="gap-6">
        <TabsList aria-label="Protocol details">
          <TabsTrigger value="performance">Model performance</TabsTrigger>
          <TabsTrigger value="compounds">Test compounds</TabsTrigger>
          <TabsTrigger value="chemistry">Chemical space</TabsTrigger>
          <TabsTrigger value="history">Run history</TabsTrigger>
        </TabsList>
        {(["performance", "compounds"] as const).map((view) => (
          <TabsContent key={view} value={view}>
            {scorecard.isLoading && <Skeleton className="h-64 w-full" />}
            {scorecard.isError && (
              <p className="text-sm text-destructive">Could not load scorecard</p>
            )}
            {scorecard.data && (
              <Scorecards
                key={protocol.id}
                scorecards={scorecard.data}
                protocolId={protocol.id}
                view={view}
                selectedTarget={selectedTarget ?? scorecard.data[0]?.target}
                onTargetChange={setSelectedTarget}
              />
            )}
          </TabsContent>
        ))}
        <TabsContent value="chemistry">
          <ProtocolChemicalSpace protocolId={protocol.id} />
        </TabsContent>
        <TabsContent value="history">
          <ProtocolRuns protocolId={protocol.id} />
        </TabsContent>
      </Tabs>

      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Publish {protocol.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              Publishing permanently locks this protocol's weights, dataset, split and metrics so
              that it can be cited. Anyone in this workspace will be able to run it. This cannot be
              undone.
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
