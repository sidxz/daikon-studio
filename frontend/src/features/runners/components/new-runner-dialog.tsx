"use client";

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
import { getApiBaseUrl } from "@/shared/lib/api/custom-instance";
import { useAppConfig } from "@/shared/lib/app-config";
import { showSuccess } from "@/shared/lib/toast";
import { Check, Copy, Plus } from "lucide-react";
import { useState } from "react";
import { useCreateRunner } from "../hooks/use-runners";
import { type CreatedRunner, KNOWN_LANES, LANE_LABELS } from "../types";

/**
 * `getApiBaseUrl()` may carry a path in some deployments; the runner wants only the
 * origin. Must never throw -- a relative base (e.g. `APP_API_BASE_URL=/api`, the case
 * this comment used to only warn about) fails `new URL()` with no second argument, and
 * this runs at render during the one-time token reveal below: an uncaught throw here
 * replaces the whole page with the error boundary and loses a token that lives only in
 * component state (Important 5, final review). Fall back to the page's own origin,
 * which is what a relative base resolves against anyway.
 */
export function apiOrigin(): string {
  const base = getApiBaseUrl();
  try {
    return new URL(base).origin;
  } catch {
    return typeof window !== "undefined" ? window.location.origin : base;
  }
}

/** `images` comes from runtime config (`APP_RUNNER_IMAGE`, `APP_RUNNER_GPU_IMAGE`). */
export function runCommand(created: CreatedRunner, images: { cpu: string; gpu: string }): string {
  const gpu = created.lanes.includes("gpu");
  const image = gpu ? images.gpu : images.cpu;
  return [
    "docker run -d --restart unless-stopped \\",
    ...(gpu ? ["  --gpus all \\"] : []),
    `  -e STUDIO_URL=${apiOrigin()} \\`,
    `  -e STUDIO_RUNNER_TOKEN=${created.token} \\`,
    // The cpu image's default CMD serves the API, not the runner agent (only
    // Dockerfile.gpu's CMD is the agent already) -- override it explicitly so the
    // command this dialog hands out actually starts a runner (Critical 2, final review).
    ...(gpu ? [`  ${image}`] : [`  ${image} \\`, "  python -m daikonstudio.infrastructure.runner"]),
  ].join("\n");
}

const DEFAULT_LANES: string[] = ["default"];

export function NewRunnerDialog() {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [lanes, setLanes] = useState<string[]>(DEFAULT_LANES);
  const [created, setCreated] = useState<CreatedRunner | null>(null);
  const [copied, setCopied] = useState(false);
  const createRunner = useCreateRunner();
  const { runnerImage, runnerGpuImage } = useAppConfig();
  const images = { cpu: runnerImage, gpu: runnerGpuImage };

  const revealing = created !== null;

  function reset() {
    setName("");
    setLanes(DEFAULT_LANES);
    setCreated(null);
    setCopied(false);
    createRunner.reset();
  }

  function toggleLane(lane: string) {
    setLanes((current) =>
      current.includes(lane) ? current.filter((item) => item !== lane) : [...current, lane],
    );
  }

  async function copyCommand() {
    if (!created) return;
    await navigator.clipboard.writeText(runCommand(created, images));
    setCopied(true);
    showSuccess("Copied to clipboard");
  }

  return (
    <>
      <Button onClick={() => setOpen(true)}>
        <Plus className="size-4" />
        Add runner
      </Button>

      <Dialog
        open={open}
        onOpenChange={(next) => {
          // Closing during the reveal step loses the token for good -- only the
          // explicit "I've copied it" button below is allowed to do that.
          if (!next && revealing) return;
          setOpen(next);
          if (!next) reset();
        }}
      >
        <DialogContent
          showCloseButton={!revealing}
          onEscapeKeyDown={(event) => revealing && event.preventDefault()}
          onInteractOutside={(event) => revealing && event.preventDefault()}
        >
          {!created ? (
            <>
              <DialogHeader>
                <DialogTitle>Add a runner</DialogTitle>
                <DialogDescription>
                  Register a machine to run training on your own hardware. You'll get a one-time
                  token to start it with.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="runner-name">Name</Label>
                  <Input
                    id="runner-name"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="e.g. lab-workstation-1"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Lanes</Label>
                  <div className="flex flex-wrap gap-1.5">
                    {KNOWN_LANES.map((lane) => (
                      <Button
                        key={lane}
                        type="button"
                        variant={lanes.includes(lane) ? "secondary" : "outline"}
                        size="sm"
                        onClick={() => toggleLane(lane)}
                      >
                        {LANE_LABELS[lane] ?? lane}
                      </Button>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    The queues this runner accepts jobs from.
                  </p>
                </div>
              </div>
              <DialogFooter className="gap-2">
                <Button variant="outline" onClick={() => setOpen(false)}>
                  Cancel
                </Button>
                <Button
                  disabled={!name.trim() || lanes.length === 0 || createRunner.isPending}
                  onClick={() =>
                    createRunner.mutate(
                      { name: name.trim(), lanes },
                      { onSuccess: (runner) => setCreated(runner) },
                    )
                  }
                >
                  {createRunner.isPending ? "Adding…" : "Add runner"}
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>{created.name} is registered</DialogTitle>
                <DialogDescription>
                  This token is shown once. Treat it like a password. If you lose it, revoke this
                  runner and register a new one.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-2">
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
                  {runCommand(created, images)}
                </pre>
                <Button variant="outline" size="sm" onClick={copyCommand}>
                  {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                  {copied ? "Copied" : "Copy command"}
                </Button>
                <p className="text-xs text-muted-foreground">
                  STUDIO_URL is set to the API address this browser uses. If the runner machine
                  cannot reach it (for example, a local address or a firewall), replace it with one
                  that it can.
                  {created.lanes.includes("gpu") && (
                    <>
                      {" "}
                      The GPU image is not published by CI. Build it on the runner machine with{" "}
                      <code>make image-runner-gpu</code>.
                    </>
                  )}
                </p>
              </div>
              <DialogFooter>
                <Button
                  onClick={() => {
                    setOpen(false);
                    reset();
                  }}
                >
                  I have copied the command
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
