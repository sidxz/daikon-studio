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
import { Button, buttonVariants } from "@/shared/components/ui/button";
import type { ProtocolResponse } from "@/shared/lib/api/model";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useDeleteProtocol } from "../hooks/use-protocols";

export function DeleteProtocolButton({ protocol }: { protocol: ProtocolResponse }) {
  const router = useRouter();
  const remove = useDeleteProtocol();
  const [open, setOpen] = useState(false);

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
            <AlertDialogTitle>Delete draft protocol “{protocol.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes the trained model, its scorecard and chemical-space map, and
              the training run that produced it. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {remove.error && (
            <p role="alert" className="text-sm text-destructive">
              {remove.error.message}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: "destructive" })}
              disabled={remove.isPending}
              onClick={(event) => {
                // Radix would close before the mutation resolves.
                event.preventDefault();
                remove.mutate(protocol.id, {
                  onSuccess: () => {
                    setOpen(false);
                    router.push("/protocols");
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
