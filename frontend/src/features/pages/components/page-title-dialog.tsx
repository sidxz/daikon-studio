"use client";

import { type FormEvent, useId, useState } from "react";

import { Button } from "@/shared/components/ui/button";
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

/** One title field in a dialog: naming a new page, or renaming one. Every open
 *  is a fresh mount, so the field seeds from `initial` without an effect. */
export function PageTitleDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  initial = "",
  pending,
  onSubmit,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel: string;
  initial?: string;
  pending: boolean;
  onSubmit: (title: string) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        {open && (
          <TitleForm
            title={title}
            description={description}
            confirmLabel={confirmLabel}
            initial={initial}
            pending={pending}
            onSubmit={onSubmit}
            onCancel={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function TitleForm({
  title,
  description,
  confirmLabel,
  initial,
  pending,
  onSubmit,
  onCancel,
}: {
  title: string;
  description: string;
  confirmLabel: string;
  initial: string;
  pending: boolean;
  onSubmit: (title: string) => void;
  onCancel: () => void;
}) {
  const id = useId();
  const [value, setValue] = useState(initial);
  const trimmed = value.trim();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (trimmed && !pending) onSubmit(trimmed);
  };
  return (
    <form onSubmit={submit} className="grid gap-4">
      <DialogHeader>
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      <div className="grid gap-2">
        <Label htmlFor={id}>Title</Label>
        <Input
          id={id}
          value={value}
          maxLength={200}
          autoFocus
          placeholder="e.g. Assay conditions, 7 Oct"
          onChange={(e) => setValue(e.target.value)}
        />
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" disabled={!trimmed || pending}>
          {confirmLabel}
        </Button>
      </DialogFooter>
    </form>
  );
}
