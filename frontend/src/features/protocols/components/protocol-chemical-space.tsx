"use client";

import {
  ChemicalSpaceMap,
  type MapHit,
  useMapColors,
} from "@/shared/components/chemical-space/chemical-space-map";
import { MapLegend } from "@/shared/components/chemical-space/legend";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useMemo } from "react";
import { useProtocolChemicalSpace, useProtocolMapCompound } from "../hooks/use-protocols";

const PARTITION_LABEL = { train: "Training", validation: "Validation", test: "Test" } as const;

/** Hover content for one compound of a protocol's map. Shared with the run page. */
export function MapCompoundTooltip({ protocolId, index }: { protocolId: string; index: number }) {
  const { data, isLoading } = useProtocolMapCompound(protocolId, index);
  if (isLoading) return <p className="text-muted-foreground">Loading…</p>;
  if (!data) return <p className="text-muted-foreground">Could not load this compound</p>;
  return (
    <div className="space-y-1.5">
      <StructureThumbnail smiles={data.structure} size={96} />
      {data.compound_id && <p className="font-mono font-medium">{data.compound_id}</p>}
      <p className="font-medium">{PARTITION_LABEL[data.partition]} compound</p>
    </div>
  );
}

/**
 * The dataset behind a protocol, laid out by structure: where the training,
 * validation and test compounds sit relative to one another. On a scaffold
 * split the test compounds form their own regions; on a random split they are
 * scattered among the training compounds.
 */
export function ProtocolChemicalSpace({ protocolId }: { protocolId: string }) {
  const map = useProtocolChemicalSpace(protocolId);
  const colors = useMapColors();
  const points = map.data?.points ?? null;
  const counts = map.data?.counts ?? null;

  const base = useMemo(
    () => points && { x: points.x, y: points.y, style: points.partition },
    [points],
  );

  const summary = counts
    ? `${(counts.train ?? 0).toLocaleString()} training, ${(counts.validation ?? 0).toLocaleString()} validation and ${(counts.test ?? 0).toLocaleString()} test compounds.`
    : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Chemical space</CardTitle>
        {summary && <p className="text-sm text-muted-foreground">{summary}</p>}
      </CardHeader>
      <CardContent className="space-y-3">
        {map.isLoading && <Skeleton className="h-[420px] w-full" />}
        {map.isError && <p className="text-sm text-destructive">Could not load the map</p>}
        {map.data?.status === "missing" && (
          <p className="text-sm text-muted-foreground">
            No map has been computed for this protocol yet.
          </p>
        )}
        {base && (
          <>
            <ChemicalSpaceMap
              base={base}
              colors={colors}
              label={`Chemical space map: ${summary ?? ""}`}
              pickBase
              tooltip={(hit: MapHit) => (
                <MapCompoundTooltip protocolId={protocolId} index={hit.index} />
              )}
            />
            <MapLegend
              items={[
                { swatch: "train", label: "Training" },
                { swatch: "validation", label: "Validation" },
                { swatch: "test", label: "Test" },
              ]}
            />
            <p className="text-xs text-muted-foreground">
              Each dot is a compound in this protocol's dataset, placed by UMAP so that similar
              structures sit together. Distance on the map is approximate.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
