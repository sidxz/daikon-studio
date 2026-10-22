/**
 * What would count as a result, and when the plan would stop instead. Shared by
 * the plan being drafted and the study it became, so the two read the same.
 */
export function PlanTerms({
  checks,
  stopConditions,
}: {
  checks: string[];
  stopConditions: string[];
}) {
  return (
    <>
      {checks.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium">What would count as a result</h2>
          <p className="text-xs text-muted-foreground">
            Written before anything runs, so the answer cannot be decided after seeing it.
          </p>
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
    </>
  );
}
