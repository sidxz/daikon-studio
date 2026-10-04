import type { DatasetResponse } from "@/shared/lib/api/model";
import { type Engine, jointEnginesRefused } from "../types";

function separator(index: number, count: number): string {
  if (index === 0) return "";
  if (index < count - 1) return ", ";
  return count > 2 ? ", and " : " and ";
}

/**
 * What a training request on this dataset will predict, and -- when its targets
 * mix kinds -- which joint engines are not offered and why, so a missing option
 * reads as a rule rather than a bug.
 */
export function TargetsHint({ dataset, engines }: { dataset: DatasetResponse; engines: Engine[] }) {
  const { targets } = dataset;
  const refused = jointEnginesRefused(engines, targets);
  return (
    <div className="space-y-1 text-xs text-muted-foreground">
      <p>
        Predicting{" "}
        {targets.map((target, index) => (
          <span key={target.column}>
            {separator(index, targets.length)}
            <span className="font-mono">{target.column}</span>
          </span>
        ))}
        {targets.length === 1 && targets[0].unit && (
          <>
            {" "}
            in <span className="font-mono">{targets[0].unit}</span>
          </>
        )}
        , held out by {dataset.split.strategy} split.
      </p>
      {refused.length > 0 && (
        <p>
          {new Intl.ListFormat("en-US", { type: "conjunction" }).format(
            refused.map((engine) => engine.name),
          )}{" "}
          {refused.length === 1
            ? "trains one joint model and needs targets of a single kind, so it is"
            : "train one joint model and need targets of a single kind, so they are"}{" "}
          not offered for this dataset.
        </p>
      )}
    </div>
  );
}
