"use client";

import { useCreateCollection } from "@/features/collections";
import { useProtocol } from "@/features/protocols";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
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
import { useBreadcrumbTrail } from "@/shared/lib/stores/breadcrumb-store";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { useCancelRun, useRun } from "../hooks/use-runs";
import { RUN_STATUS_COPY } from "../types";
import { TriageGrid } from "./triage-grid";

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
  const { data: run, isLoading } = useRun(runId);
  const { data: protocol } = useProtocol(run?.protocol_id ?? undefined);
  const cancel = useCancelRun();
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

  if (isLoading || !run) {
    return (
      <div className="mx-auto w-full max-w-6xl space-y-4 p-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const running = run.status === "pending" || run.status === "running";

  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 p-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">{protocol?.name ?? "Prediction run"}</h1>
            <Badge variant={run.status === "ready" ? "default" : "outline"} className="font-normal">
              {RUN_STATUS_COPY[run.status] ?? run.status}
            </Badge>
          </div>
          {/* Identified by protocol and date, never by id -- a chemist says
              "the July 29th run". */}
          <p className="mt-1 text-sm text-muted-foreground">
            {new Date(run.created_at).toLocaleString()}
          </p>
          {hasSubmittedCount && (
            <p className="mt-1 text-sm text-muted-foreground">
              {submittedCount} compound{submittedCount === 1 ? "" : "s"} submitted
            </p>
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
      </div>

      {running && (
        <Card>
          <CardContent className="space-y-3 py-6">
            <p className="text-sm text-muted-foreground">{run.phase ?? "Starting…"}</p>
            <Progress value={Math.round(run.progress * 100)} />
          </CardContent>
        </Card>
      )}

      {run.status === "failed" && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">This run failed</p>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{run.error_message}</p>
        </div>
      )}

      {fromCache && run.status === "ready" && (
        <div className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
          These compounds had already been scored by this protocol — these results came from cache,
          not a new run.
        </div>
      )}

      {run.status === "ready" && protocol && (
        <TriageGrid
          runId={runId}
          readouts={protocol.readouts}
          saving={createCollection.isPending}
          onSaveSelection={(rowIds) => {
            setPendingRows(rowIds);
            setCollectionName("");
          }}
        />
      )}

      <Dialog open={pendingRows !== null} onOpenChange={(open) => !open && setPendingRows(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Save {pendingRows?.length ?? 0} compound
              {pendingRows?.length === 1 ? "" : "s"} as a collection
            </DialogTitle>
            <DialogDescription>
              A collection is a frozen copy of these rows, marked as AI-predicted. It keeps its own
              snapshot, so it survives whatever happens to this run.
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
