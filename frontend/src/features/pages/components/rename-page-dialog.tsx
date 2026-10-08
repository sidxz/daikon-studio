"use client";

import { useRetitlePage } from "@/features/pages/lib/hooks";
import type { PageView } from "@/features/pages/lib/types";
import { showSuccess } from "@/shared/lib/toast";

import { PageTitleDialog } from "./page-title-dialog";

export function RenamePageDialog({
  page,
  open,
  onOpenChange,
}: {
  page: PageView;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const retitle = useRetitlePage(page.id);
  return (
    <PageTitleDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Rename page"
      description="The new title shows everywhere this page is listed."
      confirmLabel="Rename"
      initial={page.title}
      pending={retitle.isPending}
      onSubmit={(title) =>
        retitle.mutate(title, {
          onSuccess: () => {
            onOpenChange(false);
            showSuccess("Page renamed");
          },
        })
      }
    />
  );
}
