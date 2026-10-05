"use client";

import { MoveToFolderMenu, setDragItem, useFolders } from "@/features/folders";
import { ItemCard, LeadNumber } from "@/shared/components/item-card";
import type { Headline } from "@/shared/lib/headlines";
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

function Score({ headline, more }: { headline: Headline; more: number }) {
  const label = metricLabel(headline.primary_metric);
  const { value, baseline_value: baseline } = headline;
  return (
    <div className="space-y-1">
      {value === null ? (
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-sm text-muted-foreground">No score</p>
        </div>
      ) : (
        <LeadNumber
          label={label}
          value={value.toFixed(3)}
          aside={
            baseline === null ? undefined : versusBaseline(headline.primary_metric, value, baseline)
          }
        />
      )}
      {more > 0 && (
        <p className="text-xs text-muted-foreground">
          +{more} more target{more === 1 ? "" : "s"}
        </p>
      )}
    </div>
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
  const [readout, ...otherReadouts] = protocol.readouts ?? [];
  const [headline, ...otherScores] = scores;

  return (
    <ItemCard
      href={`/protocols/${protocol.id}`}
      name={protocol.name}
      subtitle={
        readout &&
        `Predicts ${readout.name}${otherReadouts.length ? ` and ${otherReadouts.length} more` : ""}`
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
      createdAt={protocol.created_at}
      draggable={canEdit}
      onDragStart={(e) => setDragItem(e, "protocol", protocol.id)}
    >
      {headline && <Score headline={headline} more={otherScores.length} />}
    </ItemCard>
  );
}
