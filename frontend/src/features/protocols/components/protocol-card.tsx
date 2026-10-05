"use client";

import { MoveToFolderMenu, setDragItem, useFolders } from "@/features/folders";
import { ItemCard, LeadNumber } from "@/shared/components/item-card";
import type { Headline } from "@/shared/lib/headlines";
import { targetsOf } from "@/shared/lib/targets";
import { higherIsBetter } from "../lib/verdict";
import { type Protocol, metricLabel } from "../types";

/** Below this the model and its baseline score the same. */
const TIE = 1e-9;

/**
 * "Baseline 0.588" over "0.015 better", signed by the metric's direction. No color:
 * the detail page owns the noise-aware verdict, and a raw margin is not one.
 */
function versusBaseline(metric: string, value: number, baseline: number) {
  const delta = higherIsBetter(metric) ? value - baseline : baseline - value;
  const change =
    Math.abs(delta) < TIE
      ? "Same as baseline"
      : `${Math.abs(delta).toFixed(3)} ${delta > 0 ? "better" : "worse"}`;
  return (
    <>
      <span className="block">Baseline {baseline.toFixed(3)}</span>
      <span className="block">{change}</span>
    </>
  );
}

/** With several targets the label names the one scored: "MCC on aggregator". */
function Score({ headline, several }: { headline: Headline; several: boolean }) {
  const metric = metricLabel(headline.primary_metric);
  const label = several ? `${metric} on ${headline.column}` : metric;
  const { value, baseline_value: baseline } = headline;
  if (value === null) {
    return (
      <div>
        <p className="truncate text-xs text-muted-foreground">{label}</p>
        <p className="text-sm text-muted-foreground">No score</p>
      </div>
    );
  }
  return (
    <LeadNumber
      label={label}
      value={value.toFixed(3)}
      aside={
        baseline === null ? undefined : versusBaseline(headline.primary_metric, value, baseline)
      }
    />
  );
}

export function ProtocolCard({
  protocol,
  scores,
  engineName,
  creator,
}: {
  protocol: Protocol;
  /** Its training run's headlines, one per target; empty when none was found. */
  scores: Headline[];
  engineName: string;
  creator?: string;
}) {
  const canEdit = useFolders("protocol").data?.can_edit ?? false;
  // Targets, not readouts: a classifier's probability column is not a second target.
  const targets = targetsOf(protocol.readouts ?? []);

  return (
    <ItemCard
      href={`/protocols/${protocol.id}`}
      name={protocol.name}
      subtitle={
        targets.length > 0 && (
          <span title={targets.join(", ")}>
            Predicts {targets[0]}
            {targets.length > 1 && ` and ${targets.length - 1} more`}
          </span>
        )
      }
      draft={protocol.status === "draft"}
      action={
        <MoveToFolderMenu
          kind="protocol"
          itemId={protocol.id}
          currentFolderId={protocol.folder_id}
        />
      }
      footerStart={engineName}
      creator={creator}
      creatorId={protocol.created_by}
      createdAt={protocol.created_at}
      draggable={canEdit}
      onDragStart={(e) => setDragItem(e, "protocol", protocol.id)}
    >
      {scores[0] && <Score headline={scores[0]} several={targets.length > 1} />}
    </ItemCard>
  );
}
