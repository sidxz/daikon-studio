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
import { Button, buttonVariants } from "@/shared/components/ui/button";
import type { DatasetResponse } from "@/shared/lib/api/model";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useDatasetProtocols, useDeleteDataset } from "../hooks/use-datasets";

export function DeleteDatasetButton({ dataset }: { dataset: DatasetResponse }) {
  const router = useRouter();
  const remove = useDeleteDataset();
  const [open, setOpen] = useState(false);
  // Fetched when the dialog opens, so it reflects any protocol deleted since.
  const protocols = useDatasetProtocols(dataset.id, open);
  const blockers = protocols.data?.items ?? [];

  return (
    <>
      <Button
        variant="outline"
        onClick={() => {
          remove.reset();
          setOpen(true);
        }}
      >
        Delete
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{dataset.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes the dataset's frozen snapshot and profile. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {protocols.isLoading && (
            <p className="text-sm text-muted-foreground">
              Checking for protocols trained on this dataset…
            </p>
          )}
          {blockers.length > 0 && (
            <div className="space-y-2 text-sm">
              <p>
                Protocols trained on this dataset must be deleted first. A dataset used by a
                published protocol cannot be deleted.
              </p>
              <ul className="space-y-1">
                {blockers.map((protocol) => (
                  <li key={protocol.id} className="flex items-center gap-2">
                    <Link
                      href={`/protocols/${protocol.id}`}
                      className="underline underline-offset-2"
                    >
                      {protocol.name}
                    </Link>
                    <span className="text-muted-foreground">v{protocol.protocol_version}</span>
                    <Badge
                      variant={protocol.status === "draft" ? "outline" : "default"}
                      className="font-normal"
                    >
                      {protocol.status === "draft" ? "Draft" : "Published"}
                    </Badge>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {remove.error && (
            <p role="alert" className="text-sm text-destructive">
              {remove.error.message}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: "destructive" })}
              disabled={remove.isPending || protocols.isLoading || blockers.length > 0}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                remove.mutate(dataset.id, {
                  onSuccess: () => {
                    setOpen(false);
                    router.push("/datasets");
                  },
                });
              }}
            >
              {remove.isPending ? "Deleting…" : "Delete permanently"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
