"use client";

import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { rememberWorkspace, rememberedWorkspace } from "@/shared/lib/auth/workspace-memory";
import type { AuthzWorkspaceSelectorProps } from "@duar-auth/nextjs";
import { useEffect, useRef, useState } from "react";

type Decision = { kind: "pending" } | { kind: "picker" } | { kind: "auto"; id: string };

export function WorkspaceSelector({
  workspaces,
  onSelect,
  isLoading,
}: AuthzWorkspaceSelectorProps) {
  const [decision, setDecision] = useState<Decision>({ kind: "pending" });

  // StrictMode re-runs the mount effect before the state change is observable,
  // so the ref is what stops a second mint firing for the same workspace.
  const autoFiredRef = useRef(false);

  // Decided in an effect rather than a useState initializer: localStorage is
  // client-only, and reading it during render would hydrate-mismatch.
  useEffect(() => {
    if (decision.kind !== "pending" || isLoading || autoFiredRef.current) return;
    const remembered = rememberedWorkspace();
    if (remembered && workspaces.some((workspace) => workspace.id === remembered)) {
      autoFiredRef.current = true;
      setDecision({ kind: "auto", id: remembered });
      onSelect(remembered);
    } else {
      setDecision({ kind: "picker" });
    }
  }, [decision.kind, isLoading, workspaces, onSelect]);

  if (decision.kind !== "picker") {
    const workspace =
      decision.kind === "auto" ? workspaces.find((w) => w.id === decision.id) : null;
    return (
      <div>
        <h2 className="text-sm font-medium text-muted-foreground">
          {workspace ? `Entering ${workspace.name}…` : "Signing in…"}
        </h2>
        <div className="mt-4 space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-sm font-medium text-muted-foreground">Select workspace to continue</h2>
      <div className="mt-4 space-y-2">
        {workspaces.map((workspace) => (
          <Button
            key={workspace.id}
            variant="outline"
            className="w-full justify-start rounded-[11px]"
            disabled={isLoading}
            onClick={() => {
              rememberWorkspace(workspace.id);
              onSelect(workspace.id);
            }}
          >
            <span className="truncate">{workspace.name}</span>
            <span className="ml-auto text-xs text-muted-foreground">{workspace.role}</span>
          </Button>
        ))}
      </div>
    </div>
  );
}
