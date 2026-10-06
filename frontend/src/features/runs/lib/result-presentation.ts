import type { ReadoutResponse } from "@/shared/lib/api/model";
import { uncertaintyColumn } from "@/shared/lib/targets";
import type { TriageRow } from "../types";
import { IN_DOMAIN_FLOOR, type NumberFilter, boundsFor } from "./result-query";

export function resultTargets(readouts: ReadoutResponse[]) {
  const targets = readouts.filter((readout) => readout.type !== "probability");
  return targets.map((readout) => ({
    name: readout.name,
    readout,
    probability: readouts.find(
      (candidate) =>
        candidate.type === "probability" && candidate.name === `${readout.name}_probability`,
    ),
    uncertainty: uncertaintyColumn(readout.name, targets.length),
  }));
}

export function readoutDescription(readout: ReadoutResponse): string {
  return [
    readout.description,
    readout.unit && `Unit: ${readout.unit}.`,
    // A probability's direction is always "high" in the API. It describes
    // increasing P(class=1), not whether that class is desirable.
    readout.type !== "probability" &&
      readout.direction &&
      `Preferred direction: ${readout.direction === "low" ? "lower" : "higher"}.`,
    readout.type === "class" &&
      `Positive (class 1) at probability ≥ ${readout.threshold ?? 0.5}; negative (class 0) below it.`,
  ]
    .filter(Boolean)
    .join(" ");
}

export function uncertaintyInfo(engineId: string | undefined, readout: ReadoutResponse) {
  const binary = readout.type === "class";
  const unit = !binary && readout.unit ? ` In ${readout.unit}.` : "";
  if (engineId === "ecfp4-randomforest" || engineId === "tanimoto-gp") {
    if (binary)
      return {
        description:
          "Proximity to probability 0.5, from 0 to 1. This is a heuristic, not a calibrated error probability.",
        ceiling: 1,
      };
    return {
      description:
        (engineId === "tanimoto-gp"
          ? "Posterior standard deviation."
          : "Standard deviation across the forest's tree predictions.") + unit,
      ceiling: null,
    };
  }
  if (engineId === "chemprop-dmpnn")
    return {
      description: `Standard deviation across ensemble predictions. Unavailable for a single model.${unit}`,
      ceiling: binary ? 0.5 : null,
    };
  return {
    description:
      "Uncertainty reported by this model. Missing values mean no estimate was reported.",
    ceiling: null,
  };
}

/** Evaluate selected rows using the same inclusive bounds sent to the API. */
export function rowMatchesFilters(
  row: TriageRow,
  filters: Record<string, NumberFilter>,
  inDomainOnly: boolean,
  readouts: ReadoutResponse[],
): boolean {
  if (inDomainOnly && (row.applicability == null || row.applicability < IN_DOMAIN_FLOOR))
    return false;
  const targets = resultTargets(readouts);
  return Object.entries(filters).every(([column, model]) => {
    const bounds = boundsFor(model);
    if (!bounds) return true;
    const target = targets.find((target) => target.uncertainty === column);
    const value =
      column === "applicability"
        ? row.applicability
        : target
          ? row.uncertainty?.[target.name]
          : row.readouts?.[column]?.value;
    return (
      value != null &&
      Number.isFinite(value) &&
      (bounds.min == null || value >= bounds.min) &&
      (bounds.max == null || value <= bounds.max)
    );
  });
}

export function filterLabel(model: NumberFilter, isClass: boolean): string {
  if (model.type === "equals")
    return isClass ? (model.filter === 1 ? "Positive" : "Negative") : `= ${model.filter}`;
  if (model.type === "inRange") return `${model.filter}–${model.filterTo}`;
  return `${model.type === "lessThanOrEqual" ? "≤" : "≥"} ${model.filter}`;
}
