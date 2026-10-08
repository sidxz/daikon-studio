"use client";

import { ConfirmDialog } from "@/shared/components/ui/confirm-dialog";

/** Shown when a revise 409s (someone else saved first). We keep the user's text and
 *  offer to reload the server head; explicit confirm/cancel per the app UI rules. */
export function ConflictDialog({
  open,
  onReload,
  onDismiss,
}: {
  open: boolean;
  onReload: () => void;
  onDismiss: () => void;
}) {
  return (
    <ConfirmDialog
      open={open}
      onOpenChange={(o) => !o && onDismiss()}
      title="This page changed elsewhere"
      description="Another edit was saved while you were writing. Reload the latest version? Your unsaved text stays in this tab until you do."
      confirmLabel="Reload latest"
      onConfirm={onReload}
    />
  );
}
