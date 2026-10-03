"use client";

import { MapCompoundTooltip, useProtocolChemicalSpace } from "@/features/protocols";
import {
  ChemicalSpaceMap,
  type MapHit,
  useMapColors,
} from "@/shared/components/chemical-space/chemical-space-map";
import { MapLegend } from "@/shared/components/chemical-space/legend";
import { STYLE } from "@/shared/components/chemical-space/renderer";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import type { RunMapSummaryResponse } from "@/shared/lib/api/model";
import { useMemo } from "react";
import { useRunChemicalSpace, useRunMapCompound } from "../hooks/use-runs";

/** The card's one-line reading of a run, from the server's exact counts. */
export function summarySentence(summary: RunMapSummaryResponse): string {
  const noun = summary.total === 1 ? "compound is" : "compounds are";
  const head = `${summary.in_domain.toLocaleString()} of ${summary.total.toLocaleString()} ${noun} inside the applicability domain (similarity ≥ ${summary.threshold}).`;
  if (summary.nearest_min == null || summary.nearest_max == null) return head;
  const range =
    summary.nearest_min === summary.nearest_max
      ? summary.nearest_min.toFixed(2)
      : `${summary.nearest_min.toFixed(2)} to ${summary.nearest_max.toFixed(2)}`;
  return `${head} Nearest training compound: similarity ${range}.`;
}

function RunCompoundTooltip({ runId, row }: { runId: string; row: number }) {
  const { data, isLoading } = useRunMapCompound(runId, row);
  if (isLoading) return <p className="text-muted-foreground">Loading…</p>;
  if (!data) return <p className="text-muted-foreground">Could not load this compound</p>;
  return (
    <div className="space-y-1.5">
      <StructureThumbnail smiles={data.structure} size={96} />
      {data.compound_id && <p className="font-medium">{data.compound_id}</p>}
      {Object.entries(data.values).map(([name, value]) => (
        <p key={name}>
          <span className="text-muted-foreground">{name}</span>{" "}
          <ReadoutValue value={value} precision={3} />
        </p>
      ))}
      <p>
        <span className="text-muted-foreground">Similarity to nearest training compound</span>{" "}
        {data.applicability == null ? "—" : data.applicability.toFixed(2)}
      </p>
    </div>
  );
}

/**
 * Where a run's compounds sit among the protocol's training compounds.
 *
 * Position is the similarity-weighted centre of each compound's five nearest
 * training compounds, so even a compound unlike anything lands inside the cloud:
 * position answers "near which chemistry", and the marker answers "how near".
 * A dashed ring is a compound below the applicability threshold.
 */
export function RunChemicalSpace({ runId, protocolId }: { runId: string; protocolId: string }) {
  const map = useProtocolChemicalSpace(protocolId);
  const run = useRunChemicalSpace(runId);
  const colors = useMapColors();
  const mapPoints = map.data?.points ?? null;
  const runPoints = run.data?.points ?? null;
  const summary = run.data?.summary ?? null;

  // Only the training compounds: applicability, and so placement, is measured
  // against them. `trainIndex[i]` is base point i's index in the protocol map.
  const layers = useMemo(() => {
    if (!mapPoints || !runPoints || !summary) return null;
    const trainIndex: number[] = [];
    mapPoints.partition.forEach((partition, index) => {
      if (partition === 0) trainIndex.push(index);
    });
    const base = {
      x: Float32Array.from(trainIndex, (i) => mapPoints.x[i]),
      y: Float32Array.from(trainIndex, (i) => mapPoints.y[i]),
      style: new Float32Array(trainIndex.length).fill(STYLE.train),
    };
    const overlay = {
      x: runPoints.x,
      y: runPoints.y,
      style: runPoints.applicability.map((value) =>
        value != null && value >= summary.threshold ? STYLE.runIn : STYLE.runOut,
      ),
    };
    return { base, overlay, trainIndex };
  }, [mapPoints, runPoints, summary]);

  const missing = map.data?.status === "missing" || run.data?.status === "missing";
  const loading = map.isLoading || run.isLoading;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Chemical space</CardTitle>
        {summary && <p className="text-sm text-muted-foreground">{summarySentence(summary)}</p>}
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && <Skeleton className="h-[420px] w-full" />}
        {(map.isError || run.isError) && (
          <p className="text-sm text-destructive">Could not load the map</p>
        )}
        {!loading && missing && (
          <p className="text-sm text-muted-foreground">
            No map has been computed for this protocol yet.
          </p>
        )}
        {layers && runPoints && mapPoints && summary && (
          <>
            <ChemicalSpaceMap
              base={layers.base}
              overlay={layers.overlay}
              colors={colors}
              label={`Chemical space map: ${layers.trainIndex.length.toLocaleString()} training compounds and ${summarySentence(summary)}`}
              pickBase
              lines={(hit: MapHit) =>
                hit.layer === "overlay"
                  ? runPoints.neighbors[hit.index].map(
                      (i) => [mapPoints.x[i], mapPoints.y[i]] as [number, number],
                    )
                  : []
              }
              tooltip={(hit: MapHit) =>
                hit.layer === "overlay" ? (
                  <RunCompoundTooltip runId={runId} row={runPoints.row_id[hit.index]} />
                ) : (
                  <MapCompoundTooltip
                    protocolId={protocolId}
                    index={layers.trainIndex[hit.index]}
                  />
                )
              }
            />
            <MapLegend
              items={[
                { swatch: "train", label: "Training compound" },
                { swatch: "runIn", label: "This run, inside the domain" },
                { swatch: "runOut", label: "This run, outside the domain" },
              ]}
            />
            <p className="text-xs text-muted-foreground">
              Each dot is a compound, placed by UMAP so that similar structures sit together.
              Position shows which part of the training set a compound is closest to; distance on
              the map is approximate. A filled marker is inside the applicability domain; a dashed
              ring is outside it. Hover a compound to see its nearest training compounds.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
