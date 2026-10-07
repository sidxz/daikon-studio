import type { ScorecardResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { formatCutoff } from "../lib/format-cutoff";

export function ConfusionMatrix({ scorecard }: { scorecard: ScorecardResponse }) {
  const summary = scorecard.classification_summary;
  if (!summary || scorecard.prediction_kind !== "probability") return null;
  const cells = [
    { label: "Actives found", count: summary.true_positive, correct: true },
    { label: "Actives missed", count: summary.false_negative, correct: false },
    { label: "False alarms", count: summary.false_positive, correct: false },
    { label: "Inactives correctly rejected", count: summary.true_negative, correct: true },
  ];
  return (
    <section
      className="grid gap-6 rounded-lg border bg-card p-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]"
      aria-label="Confusion matrix"
    >
      <div className="min-w-0">
        <h3 className="text-base font-medium">Correct predictions and mistakes</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Confusion matrix · counts on the full test set
        </p>
        <table className="mt-4 w-full table-fixed border-separate border-spacing-1.5 text-center text-xs">
          <caption className="sr-only">
            Rows are actual classes. Columns are predicted classes.
          </caption>
          <thead>
            <tr>
              <td className="w-[24%]" />
              <th scope="col" className="pb-1 font-medium text-muted-foreground">
                Predicted active
              </th>
              <th scope="col" className="pb-1 font-medium text-muted-foreground">
                Predicted inactive
              </th>
            </tr>
          </thead>
          <tbody>
            {[0, 1].map((row) => (
              <tr key={row}>
                <th scope="row" className="pr-1 text-left font-medium text-muted-foreground">
                  Actually {row === 0 ? "active" : "inactive"}
                </th>
                {cells.slice(row * 2, row * 2 + 2).map((cell) => (
                  <td
                    key={cell.label}
                    className={cn(
                      "rounded-md border px-2 py-4",
                      cell.correct
                        ? "border-primary/20 bg-primary/[0.06]"
                        : "border-warning/25 bg-warning/[0.06]",
                    )}
                  >
                    <span className="block text-2xl font-semibold tabular-nums sm:text-3xl">
                      {cell.count.toLocaleString()}
                    </span>
                    <span className="mt-1.5 block leading-snug text-muted-foreground">
                      {cell.label}
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-4 self-center text-sm">
        <div>
          <h4 className="font-medium">What this means</h4>
          <p className="mt-1 leading-relaxed text-muted-foreground">
            The model found {summary.true_positive.toLocaleString()} active compounds and missed{" "}
            {summary.false_negative.toLocaleString()}. Another{" "}
            {summary.false_positive.toLocaleString()} compounds were flagged as active but were
            actually inactive.
          </p>
        </div>
        <div className="rounded-md bg-muted/50 p-3">
          <p className="font-medium">Decision cutoff: {formatCutoff(scorecard.cutoff ?? 0.5)}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            A score {summary.cutoff_inclusive ? "at or above" : "above"} this cutoff is called
            active. Precision, recall and these counts all use this same cutoff.
            {scorecard.cutoff != null
              ? " It was selected using validation data."
              : " This model uses the default cutoff."}
          </p>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          “Active” means the positive class (1) for this target. Which mistakes matter most depends
          on how you plan to use the model.
        </p>
      </div>
    </section>
  );
}
