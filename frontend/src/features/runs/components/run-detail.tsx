"use client";

import { useCreateCollection } from "@/features/collections";
import { useProtocol } from "@/features/protocols";
import { LANE_LABELS, useRunners } from "@/features/runners";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/shared/components/ui/alert-dialog";
import { Badge } from "@/shared/components/ui/badge";
import { Button, buttonVariants } from "@/shared/components/ui/button";
import { Card, CardContent } from "@/shared/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/components/ui/dialog";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import { Progress } from "@/shared/components/ui/progress";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { ApiError } from "@/shared/lib/api/custom-instance";
import type { PredictionCountsWire } from "@/shared/lib/api/model";
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useCancelRun, useRetryRun, useRun, useRunEpochs } from "../hooks/use-runs";
import { RUN_STATUS_COPY } from "../types";
import { RunChemicalSpace } from "./run-chemical-space";
import { TrainingProgress } from "./training-progress";
import { TriageGrid } from "./triage-grid";

/**
 * Mounted only while a run is queued, so the Runners poll stops with it. Says
 * nothing until that list loads, or if it fails: no hint beats a wrong one.
 */
function LaneHint({ lane }: { lane: string }) {
  const { data: runners } = useRunners();
  if (!runners) return null;
  if (runners.some((runner) => runner.online && !runner.revoked && runner.lanes.includes(lane))) {
    return null;
  }
  return (
    <p className="text-sm text-warning">
      Waiting for a runner on the "{LANE_LABELS[lane] ?? lane}" lane. None is currently online.
    </p>
  );
}

export function RunDetail({ runId }: { runId: string }) {
  const router = useRouter();
  const params = useSearchParams();
  // A URL parameter is user input, not a measurement -- `?compounds=9999`
  // must not render as fact. Parsed and range-checked before it is trusted
  // enough to show; a run opened with no parameter (or a bogus one) shows no
  // line at all, same as any other absent value.
  const rawCompounds = params.get("compounds");
  const submittedCount = rawCompounds !== null ? Number.parseInt(rawCompounds, 10) : null;
  const hasSubmittedCount =
    submittedCount !== null && Number.isFinite(submittedCount) && submittedCount > 0;
  const fromCache = params.get("cached") === "1";
  // isLoadingError, not isError: a background refetch that fails with the run
  // already on screen must not replace a triage grid and its selection with an
  // error box. With data present `pollInterval` backs off instead of stopping.
  const { data: run, isLoadingError, error, refetch } = useRun(runId);
  const { data: protocol } = useProtocol(run?.protocol_id ?? undefined);
  const training = run?.kind === "training";
  const { data: epochs } = useRunEpochs(
    training ? runId : "",
    run?.status === "pending" || run?.status === "running",
  );

  // A training run that finishes while it is being watched opens its results: the
  // protocol's scorecard. Only on that transition -- a run opened after it finished
  // stays here, where its training history is.
  const watchedStatus = useRef(run?.status);
  useEffect(() => {
    const before = watchedStatus.current;
    watchedStatus.current = run?.status;
    if (
      (before === "pending" || before === "running") &&
      run?.status === "ready" &&
      run.kind === "training" &&
      run.protocol_id
    ) {
      router.push(`/protocols/${run.protocol_id}`);
    }
  }, [run?.status, run?.kind, run?.protocol_id, router]);
  const cancel = useCancelRun();
  const retry = useRetryRun();
  const createCollection = useCreateCollection();

  const [pendingRows, setPendingRows] = useState<number[] | null>(null);
  const [collectionName, setCollectionName] = useState("");

  useBreadcrumbTrail(
    run
      ? [
          { label: "Runs", href: "/runs" },
          {
            label: protocol
              ? `${protocol.name} · ${new Date(run.created_at).toLocaleDateString()}`
              : new Date(run.created_at).toLocaleString(),
          },
        ]
      : null,
  );

  // Before the skeleton: an errored query has no data either, and polling has
  // stopped (`pollInterval`), so a skeleton here would never resolve. The
  // server's message is not shown -- a 404's names the run by its UUID.
  if (isLoadingError) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <div className="mx-auto w-full max-w-6xl p-2">
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">
            {missing ? "This run does not exist in this workspace" : "Could not load this run"}
          </p>
          {!missing && (
            <Button variant="outline" size="sm" className="mt-3" onClick={() => refetch()}>
              Try again
            </Button>
          )}
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="mx-auto w-full max-w-6xl space-y-4 p-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const running = run.status === "pending" || run.status === "running";
  // A prediction run's `metrics` once READY (`Run.record_prediction_counts`);
  // a training run's hold its headline metric instead, so these stay undefined.
  const counts = run.metrics as Partial<PredictionCountsWire> | null;
  const scored = counts?.scored_rows;
  const uploaded = counts?.uploaded_rows;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 p-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">
              {protocol?.name ??
                run.name ??
                (run.kind === "training" ? "Training run" : "Prediction run")}
            </h1>
            <Badge variant={run.status === "ready" ? "default" : "outline"} className="font-normal">
              {RUN_STATUS_COPY[run.status] ?? run.status}
            </Badge>
          </div>
          {/* Identified by protocol and date, never by id -- a chemist says
              "the July 29th run". */}
          <p className="mt-1 text-sm text-muted-foreground">
            {new Date(run.created_at).toLocaleString()}
          </p>
          {/* 7: the backend's SAVED_PROGRESS_DAYS (discard_abandoned_progress.py). */}
          {run.kind === "training" && (run.status === "failed" || run.status === "cancelled") && (
            <p className="mt-1 text-sm text-muted-foreground">
              Resume continues from the last save for 7 days after the run stopped. After that, it
              trains from the beginning.
            </p>
          )}
          {/* The server's count, once there is one, supersedes the wizard's
              client-side parse: it includes the rows that did not parse. */}
          {scored != null && uploaded != null ? (
            <p className="mt-1 text-sm text-muted-foreground">
              Predicted {scored.toLocaleString()} of {uploaded.toLocaleString()} uploaded row
              {uploaded === 1 ? "" : "s"}
              {uploaded !== scored &&
                ` · ${(uploaded - scored).toLocaleString()} could not be parsed as ${
                  uploaded - scored === 1 ? "a structure" : "structures"
                }`}
            </p>
          ) : (
            hasSubmittedCount && (
              <p className="mt-1 text-sm text-muted-foreground">
                {submittedCount} compound{submittedCount === 1 ? "" : "s"} submitted
              </p>
            )
          )}
        </div>
        {running && (
          <Button
            variant="outline"
            onClick={() => cancel.mutate(runId)}
            disabled={cancel.isPending}
          >
            Cancel run
          </Button>
        )}
        {(run.status === "failed" || run.status === "cancelled") &&
          (run.kind === "training" ? (
            <>
              <Button onClick={() => retry.mutate({ id: runId })} disabled={retry.isPending}>
                Resume
              </Button>
              {/* Confirmed: it discards saved progress, which for a long fit is hours
                  of compute that cannot be recovered. A failure is toasted by the
                  mutation cache, so the dialog does not wait on the request. */}
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="ghost" disabled={retry.isPending}>
                    Start over
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Start this run over?</AlertDialogTitle>
                    <AlertDialogDescription>
                      Its saved progress is discarded and training begins again from the start.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className={buttonVariants({ variant: "destructive" })}
                      onClick={() => retry.mutate({ id: runId, fresh: true })}
                    >
                      Start over
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          ) : (
            <Button
              variant="outline"
              onClick={() => retry.mutate({ id: runId })}
              disabled={retry.isPending}
            >
              Retry
            </Button>
          ))}
      </div>

      {running &&
        (epochs && epochs.length > 0 ? (
          // A neural fit reports each epoch: its live charts replace the bare bar.
          <TrainingProgress points={epochs} live phase={run.phase} progress={run.progress} />
        ) : (
          <Card>
            <CardContent className="space-y-3 py-6">
              <p className="text-sm text-muted-foreground">{run.phase ?? "Starting…"}</p>
              <Progress value={Math.round(run.progress * 100)} />
              {run.status === "pending" && run.lane && <LaneHint lane={run.lane} />}
            </CardContent>
          </Card>
        ))}

      {run.status === "failed" && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">This run failed</p>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{run.error_message}</p>
        </div>
      )}

      {fromCache && run.status === "ready" && (
        <div className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
          These compounds were previously predicted with this protocol. Results were loaded from the
          cache; no new run was started.
        </div>
      )}

      {/* A training run has no results to triage -- its outcome is a Protocol,
          and asking for its results is a guaranteed 404. */}
      {run.kind === "training" && !running && (
        <Card>
          <CardContent className="text-sm">
            <p className="font-medium">This is a training run</p>
            <p className="mt-1 text-muted-foreground">
              {run.protocol_id ? (
                <>
                  Its scorecard is on the{" "}
                  <Link
                    href={`/protocols/${run.protocol_id}`}
                    className="underline underline-offset-2"
                  >
                    protocol page
                  </Link>
                  .
                </>
              ) : (
                "No protocol was produced."
              )}
            </p>
          </CardContent>
        </Card>
      )}

      {run.kind === "training" && !running && epochs && epochs.length > 0 && (
        <TrainingProgress points={epochs} live={false} />
      )}

      {run.kind === "prediction" && run.status === "ready" && protocol && (
        <>
          <RunChemicalSpace runId={runId} protocolId={protocol.id} />
          <TriageGrid
            runId={runId}
            readouts={protocol.readouts}
            exportName={`${protocol.name} predictions ${run.created_at.slice(0, 10)}`}
            saving={createCollection.isPending}
            onSaveSelection={(rowIds) => {
              setPendingRows(rowIds);
              setCollectionName("");
            }}
          />
        </>
      )}

      <Dialog open={pendingRows !== null} onOpenChange={(open) => !open && setPendingRows(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Save {pendingRows?.length ?? 0} compound
              {pendingRows?.length === 1 ? "" : "s"} as a collection
            </DialogTitle>
            <DialogDescription>
              A collection stores a fixed copy of these rows, labeled AI-predicted. It is stored
              independently of this run.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="collection-name">Name</Label>
            <Input
              id="collection-name"
              value={collectionName}
              onChange={(event) => setCollectionName(event.target.value)}
              placeholder="e.g. Solubility hits for synthesis"
            />
          </div>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setPendingRows(null)}>
              Cancel
            </Button>
            <Button
              disabled={!collectionName.trim() || createCollection.isPending}
              onClick={() =>
                createCollection.mutate(
                  {
                    name: collectionName.trim(),
                    run_id: runId,
                    row_ids: pendingRows ?? [],
                  },
                  {
                    onSuccess: (collection) => {
                      setPendingRows(null);
                      router.push(`/collections/${collection.id}`);
                    },
                  },
                )
              }
            >
              {createCollection.isPending ? "Saving…" : "Save collection"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
