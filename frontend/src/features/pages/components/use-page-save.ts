"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import type { PMDoc } from "@/features/pages/lib/types";
import { ApiError } from "@/shared/lib/api/custom-instance";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

/** Manual save for the page editor: edits mark the doc dirty and nothing is
 *  written until `saveNow` runs (the Save button, or Cmd/Ctrl+S). Saving is never
 *  implicit — no debounce, no flush on unmount — so a beforeunload guard is what
 *  stops the tab closing on unsaved work. 409s surface as `conflict`.
 *  Generic: it knows nothing about page ids/versions — the caller's `save` closure
 *  carries whatever version/id context it needs (see page-container.tsx, which reads
 *  the latest version from the query cache at save time, not from a render closure). */
export function usePageSave(opts: { save: (doc: PMDoc) => Promise<void> }) {
  const [status, setStatus] = useState<SaveStatus>("idle");
  const [dirty, setDirty] = useState(false);
  const [conflict, setConflict] = useState(false);
  const pending = useRef<PMDoc | null>(null);
  // Forwarded through a ref so `saveNow` keeps a stable identity even though the
  // caller passes a new `save` closure every render (page-container.tsx's is an
  // inline arrow). The Cmd+S effect lists saveNow as a dependency, so an unstable
  // identity would re-bind the keydown listener on every keystroke.
  const saveRef = useRef(opts.save);
  useEffect(() => {
    saveRef.current = opts.save;
  });

  const onChange = useCallback((doc: PMDoc) => {
    pending.current = doc;
    setDirty(true);
  }, []);

  const saveNow = useCallback(async (): Promise<boolean> => {
    const doc = pending.current;
    if (doc === null) return true;
    pending.current = null;
    setStatus("saving");
    try {
      await saveRef.current(doc);
      setStatus("saved");
      if (pending.current === null) setDirty(false); // unless new edits raced the save
      return true;
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) setConflict(true);
      // 423 = archived mid-session: retrying can't succeed, say why instead.
      else if (err instanceof ApiError && err.status === 423)
        toast.error(err.message || "The page is archived — restore it to make changes");
      setStatus("error");
      pending.current ??= doc; // keep the doc so Save retries without another edit
      return false;
    }
  }, []);

  const discard = useCallback(() => {
    pending.current = null;
    setDirty(false);
    setStatus("idle");
  }, []);

  useEffect(() => {
    if (!dirty) return;
    // Guard on the ref, not just listener attachment: discard() nulls the ref
    // synchronously, but the [dirty] cleanup that detaches this listener is a
    // passive effect — a reload issued in the same tick (ConflictDialog's
    // onReload) would otherwise race it and still show the native prompt.
    const warn = (e: BeforeUnloadEvent) => {
      if (pending.current !== null) e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
    // ponytail: client-side route changes bypass beforeunload (Next's App Router
    // exposes no cancelable navigation event); add a router guard if losing edits
    // via in-app nav turns out to bite.
  }, [dirty]);

  return {
    onChange,
    saveNow,
    discard,
    dirty,
    status,
    conflict,
    resolveConflict: () => setConflict(false),
  };
}
