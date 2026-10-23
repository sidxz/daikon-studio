/**
 * What would count as a result, and when the plan would stop instead, as the study holds them.
 * A change made after Start is marked with how far the study had got, since a criterion
 * rewritten after a result was seen is not the criterion the plan was written against.
 */
export function PlanTerms({
  checks,
  stopConditions,
  amendedAfterSteps = null,
}: {
  checks: string[];
  stopConditions: string[];
  /** How many steps had run when the checks or stop conditions last changed; null if never. */
  amendedAfterSteps?: number | null;
}) {
  return (
    <>
      {checks.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium">What would count as a result</h2>
          {amendedAfterSteps === null && (
            <p className="text-xs text-muted-foreground">
              Written before anything runs, so the answer cannot be decided after seeing it.
            </p>
          )}
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {checks.map((check) => (
              <li key={check}>{check}</li>
            ))}
          </ul>
        </section>
      )}

      {stopConditions.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium">When it would stop instead</h2>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {stopConditions.map((condition) => (
              <li key={condition}>{condition}</li>
            ))}
          </ul>
        </section>
      )}

      {amendedAfterSteps !== null && (
        <p className="text-xs text-muted-foreground">
          {amendedAfterSteps === 0
            ? "Amended before any step ran."
            : `Amended after ${amendedAfterSteps} ${amendedAfterSteps === 1 ? "step" : "steps"} had run.`}
        </p>
      )}
    </>
  );
}
