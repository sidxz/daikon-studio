"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/shared/components/ui/card";
import { ArrowRight } from "lucide-react";
import type { Tool } from "../types";
import { SURFACE_COPY, slotLabel } from "../types";

/**
 * One tool. The slot row is the part that is not decoration: it says what the
 * tool consumes and what it gives back, which is how these compose into an
 * analysis and the reason a reader can tell which tools can follow which.
 */
export function ToolCard({ tool }: { tool: Tool }) {
  const surface = SURFACE_COPY[tool.surface];
  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h2 className="text-base font-medium leading-snug">{tool.name}</h2>
          <div className="flex flex-wrap gap-1.5">
            {tool.judgment && (
              <Badge variant="default" title={surface?.detail}>
                Needs a chemist
              </Badge>
            )}
            {!tool.judgment && surface && (
              <Badge variant="secondary" title={surface.detail}>
                {surface.title}
              </Badge>
            )}
            {tool.external && (
              <Badge variant="outline" title="Runs outside Studio, so it can be unavailable">
                External
              </Badge>
            )}
          </div>
        </div>
        <p className="text-sm text-muted-foreground">{tool.description}</p>
      </CardHeader>

      <CardContent className="mt-auto space-y-3">
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {tool.inputs.length === 0 ? (
            <span className="text-muted-foreground">Starts from nothing</span>
          ) : (
            tool.inputs.map((slot) => (
              <span
                key={slot.key}
                className="rounded border px-1.5 py-0.5 text-muted-foreground"
                title={slot.optional ? "Optional: it works without this" : undefined}
              >
                {slotLabel(slot)}
                {slot.optional && <span aria-label=", optional"> ?</span>}
              </span>
            ))
          )}
          <ArrowRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          {tool.outputs.map((slot) => (
            <span key={slot.key} className="rounded bg-muted px-1.5 py-0.5 font-medium">
              {slotLabel(slot)}
            </span>
          ))}
        </div>

        {tool.resolves_to && (
          <p className="text-xs text-muted-foreground">
            Approving one sets the{" "}
            <span className="font-medium text-foreground">
              {slotLabel({
                key: "resolved",
                kind: tool.resolves_to,
                label: "",
                optional: false,
              }).toLowerCase()}
            </span>{" "}
            for everything that follows. Nothing computes it for you.
          </p>
        )}

        <p className="text-xs text-muted-foreground">
          <span className="sr-only">Does the work: </span>
          <span aria-hidden="true">Uses </span>
          {tool.library}
        </p>
      </CardContent>
    </Card>
  );
}
