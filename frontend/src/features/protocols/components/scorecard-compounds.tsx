"use client";

import type { ScorecardResponse } from "@/shared/lib/api/model";
import { LargestErrors } from "./largest-errors";
import { RankedTestCompounds } from "./ranked-test-compounds";

export function ScorecardCompounds({ scorecard }: { scorecard: ScorecardResponse }) {
  if (
    !scorecard.ranked_high?.length &&
    !scorecard.ranked_low?.length &&
    !scorecard.worst_rows.length
  ) {
    return (
      <p className="text-sm text-muted-foreground">
        No test-compound predictions are available for this target.
      </p>
    );
  }
  return (
    <div className="space-y-6">
      <RankedTestCompounds key={scorecard.target} scorecard={scorecard} />
      <LargestErrors key={`errors-${scorecard.target}`} scorecard={scorecard} />
    </div>
  );
}
