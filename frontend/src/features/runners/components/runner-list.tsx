"use client";

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
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import Link from "next/link";
import { useState } from "react";
import { useRevokeRunner, useRunners } from "../hooks/use-runners";
import { formatLastSeen } from "../lib/format-last-seen";
import { LANE_LABELS, type Runner } from "../types";
import { NewRunnerDialog } from "./new-runner-dialog";

/**
 * Online is the one state worth a heartbeat -- offline and revoked are just
 * flat facts. The ping is the page's only ambient motion, on purpose.
 */
function StatusDot({ runner }: { runner: Runner }) {
  if (runner.revoked) {
    return <span className="size-2 shrink-0 rounded-full bg-destructive" aria-hidden />;
  }
  if (runner.online) {
    return (
      <span className="relative inline-flex size-2 shrink-0" aria-hidden>
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-success opacity-75" />
        <span className="relative inline-flex size-2 rounded-full bg-success" />
      </span>
    );
  }
  return <span className="size-2 shrink-0 rounded-full bg-muted-foreground/40" aria-hidden />;
}

function statusLabel(runner: Runner): string {
  if (runner.revoked) return "Revoked";
  return runner.online ? "Online" : "Offline";
}

export function RunnerList() {
  const { data, isLoading, isError } = useRunners();
  const revokeRunner = useRevokeRunner();
  const [revokeTarget, setRevokeTarget] = useState<Runner | null>(null);
  const runners = data ?? [];

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Runners</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Machines that run training on your own hardware, outside the hosted queue.
          </p>
        </div>
        <NewRunnerDialog />
      </div>

      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load runners</p>
        </div>
      )}

      {data && runners.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <p className="text-sm font-medium">No runners yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Add one to run training on your own hardware.
          </p>
        </div>
      )}

      {runners.length > 0 && (
        <div className="rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Lanes</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Last seen</TableHead>
                <TableHead>Current run</TableHead>
                <TableHead className="text-right">
                  <span className="sr-only">Actions</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runners.map((runner) => (
                <TableRow key={runner.id}>
                  <TableCell className="font-medium">{runner.name}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {runner.lanes.map((lane) => (
                        <Badge key={lane} variant="outline" className="font-normal">
                          {LANE_LABELS[lane] ?? lane}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className="flex items-center gap-2">
                      <StatusDot runner={runner} />
                      {statusLabel(runner)}
                    </span>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {formatLastSeen(runner.last_seen_at)}
                  </TableCell>
                  <TableCell>
                    {runner.current_run_id ? (
                      <Link
                        href={`/runs/${runner.current_run_id}`}
                        className="text-primary hover:underline"
                      >
                        View run
                      </Link>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    {!runner.revoked && (
                      <Button variant="ghost" size="sm" onClick={() => setRevokeTarget(runner)}>
                        Revoke
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <AlertDialog
        open={revokeTarget !== null}
        onOpenChange={(open) => !open && setRevokeTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Revoke {revokeTarget?.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              This runner's token stops working immediately. It will not pick up new work, and
              cannot be reconnected -- register a new runner if you need this hardware again.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={revokeRunner.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={revokeRunner.isPending}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                if (!revokeTarget) return;
                revokeRunner.mutate(revokeTarget.id, {
                  onSuccess: () => setRevokeTarget(null),
                });
              }}
            >
              {revokeRunner.isPending ? "Revoking…" : "Revoke runner"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
