"use client";

import type { ReactNode } from "react";

import { Button } from "@/shared/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/components/ui/dialog";
import { cn } from "@/shared/lib/utils";

/**
 * The shell every insert-embed dialog renders into: heading, the caller's own
 * form as children, and the explicit Cancel/Confirm pair the app UI rules
 * require. `busy` blocks *every* close path at once (Cancel, Esc, overlay
 * click, the header X — all of which route through Dialog's onOpenChange), so
 * a confirm's in-flight async work can't land on a dialog the caller has
 * already handed back.
 *
 * Dialogs don't reset their form state on close: page-editor.tsx unmounts the
 * matching dialog rather than merely hiding it, so every open is a fresh mount
 * — which is also what lets each one seed its state straight from
 * `editing.attrs`, with no prop-sync effect.
 */
export function EmbedDialogFrame({
  open,
  onClose,
  title,
  description,
  confirmLabel,
  busyLabel,
  busy = false,
  canConfirm,
  onConfirm,
  className,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description: string;
  confirmLabel: string;
  /** Replaces `confirmLabel` while `busy` — e.g. "Uploading…". */
  busyLabel?: string;
  busy?: boolean;
  canConfirm: boolean;
  onConfirm: () => void;
  className?: string;
  children: ReactNode;
}) {
  const requestClose = () => {
    if (!busy) onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && requestClose()}>
      <DialogContent className={cn("sm:max-w-md", className)}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        {children}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={requestClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="button" onClick={onConfirm} disabled={!canConfirm || busy}>
            {busy && busyLabel ? busyLabel : confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
