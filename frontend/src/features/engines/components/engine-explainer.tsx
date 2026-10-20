"use client";

import { Explainer } from "@/shared/components/explainers/explainer";
import {
  ATTENTION_CAPTION,
  ATTENTION_MS,
  ATTENTION_SEQUENCE_CAPTION,
  AttentionFigure,
  SEQUENCE_ENCODER,
} from "@/shared/components/explainers/figures/attention";
import {
  BOOSTING_CAPTION,
  BOOSTING_MS,
  BoostingFigure,
} from "@/shared/components/explainers/figures/boosting";
import {
  FOREST_CAPTION,
  FOREST_MS,
  ForestFigure,
} from "@/shared/components/explainers/figures/forest";
import {
  GP_CAPTION,
  GP_MS,
  GP_SEQUENCE_CAPTION,
  GaussianProcessFigure,
} from "@/shared/components/explainers/figures/gaussian-process";
import {
  MESSAGE_PASSING_CAPTION,
  MESSAGE_PASSING_MS,
  MessagePassingFigure,
} from "@/shared/components/explainers/figures/message-passing";
import {
  SITE_MEAN_CAPTION,
  SITE_MEAN_MS,
  SiteMeanFigure,
} from "@/shared/components/explainers/figures/site-mean";
import type { ComponentType } from "react";
import { ENGINE_FIGURE, type EngineFigure } from "../types";

const FIGURES: Record<
  EngineFigure,
  { Figure: ComponentType<{ t: number }>; ms: number; caption: string }
> = {
  forest: { Figure: ForestFigure, ms: FOREST_MS, caption: FOREST_CAPTION },
  boosting: { Figure: BoostingFigure, ms: BOOSTING_MS, caption: BOOSTING_CAPTION },
  "gaussian-process": { Figure: GaussianProcessFigure, ms: GP_MS, caption: GP_CAPTION },
  "message-passing": {
    Figure: MessagePassingFigure,
    ms: MESSAGE_PASSING_MS,
    caption: MESSAGE_PASSING_CAPTION,
  },
  attention: { Figure: AttentionFigure, ms: ATTENTION_MS, caption: ATTENTION_CAPTION },
  "gaussian-process-sequence": {
    Figure: GaussianProcessFigure,
    ms: GP_MS,
    caption: GP_SEQUENCE_CAPTION,
  },
  "site-mean": { Figure: SiteMeanFigure, ms: SITE_MEAN_MS, caption: SITE_MEAN_CAPTION },
  "attention-sequence": {
    Figure: ({ t }) => <AttentionFigure t={t} encoder={SEQUENCE_ENCODER} />,
    ms: ATTENTION_MS,
    caption: ATTENTION_SEQUENCE_CAPTION,
  },
};

export function EngineExplainer({ engineId }: { engineId: string }) {
  const kind = ENGINE_FIGURE[engineId];
  if (!kind) return null;
  const { Figure, ms, caption } = FIGURES[kind];
  return (
    <Explainer id={`engine-${kind}`} label="How it learns" durationMs={ms} caption={caption}>
      {/* Engine figures are drawn for an engine card; in a wide form they would
          scale to twice their intended size, so they keep that width. */}
      {(t) => (
        <div className="max-w-[440px]">
          <Figure t={t} />
        </div>
      )}
    </Explainer>
  );
}
