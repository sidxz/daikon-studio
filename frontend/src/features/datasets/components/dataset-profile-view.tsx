"use client";

import {
  CoverageCurveChart,
  HistogramChart,
  HorizontalBarChart,
  SplitHistogramChart,
} from "@/shared/components/charts";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import type { Dataset, DatasetProfile } from "../types";
import { descriptorLabel } from "../types";

/**
 * What the Dataset is made of, and whether the benchmark it defines is honest.
 *
 * The order is deliberate and it is the order the questions actually get asked
 * in: is the target worth modelling, is the split a real test, what chemistry is
 * in here, and would something trivial have done just as well. The split
 * integrity check comes second rather than last because it is the one that can
 * invalidate everything below it.
 */

function Stat({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: "warning" | "muted";
}) {
  return (
    <div className="min-w-[9rem] flex-1">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <div
        className={
          tone === "warning"
            ? "mt-0.5 text-xl font-semibold text-warning"
            : "mt-0.5 text-xl font-semibold"
        }
      >
        {value}
      </div>
      {detail && <p className="mt-1 text-xs text-muted-foreground">{detail}</p>}
    </div>
  );
}

function percent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function DatasetProfileSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-72" />
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

export function DatasetProfileView({
  dataset,
  profile,
}: {
  dataset: Dataset;
  profile: DatasetProfile;
}) {
  const unit = dataset.target.unit;
  const isScaffoldSplit = dataset.split.strategy === "scaffold";

  return (
    <div className="space-y-4">
      <TargetSection dataset={dataset} profile={profile} unit={unit} />
      <SplitHonestySection profile={profile} isScaffoldSplit={isScaffoldSplit} />
      <ScaffoldSection profile={profile} />
      <DescriptorSection profile={profile} />
      <CliffSection dataset={dataset} profile={profile} unit={unit} />
    </div>
  );
}

/** Is the target worth modelling, and did the split keep the two sides alike? */
function TargetSection({
  dataset,
  profile,
  unit,
}: {
  dataset: Dataset;
  profile: DatasetProfile;
  unit?: string | null;
}) {
  const distribution = profile.target_distribution;
  const balance = profile.class_balance;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What you are predicting</CardTitle>
        <p className="text-sm text-muted-foreground">
          {distribution
            ? "The measured values, train against test. A narrow range makes an impressive-looking error meaningless, and a test partition sitting somewhere else in the range is a shift the model will pay for."
            : "How the two classes fall across the partitions. A split that leaves the test set badly unbalanced makes MCC unstable, whatever the model does."}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {distribution && (
          <>
            <div className="flex flex-wrap gap-x-8 gap-y-4">
              <Stat
                label="Range"
                value={
                  <>
                    <ReadoutValue value={distribution.train.minimum} precision={2} /> to{" "}
                    <ReadoutValue value={distribution.train.maximum} unit={unit} precision={2} />
                  </>
                }
                detail="Any error has to be read against this span — the same RMSE is excellent across six log units and meaningless across half of one."
              />
              <Stat
                label="Train median"
                value={<ReadoutValue value={distribution.train.median} unit={unit} precision={2} />}
              />
              <Stat
                label="Test median"
                value={<ReadoutValue value={distribution.test.median} unit={unit} precision={2} />}
                detail={
                  Math.abs(distribution.test.median - distribution.train.median) >
                  distribution.train.std
                    ? "More than one standard deviation from the train median. The partitions are not measuring the same population."
                    : "In line with the train partition."
                }
              />
            </div>
            <SplitHistogramChart
              bins={distribution.histogram}
              xLabel={`${dataset.target.column}${unit ? ` (${unit})` : ""}`}
            />
          </>
        )}

        {balance.length > 0 && (
          <div className="space-y-3">
            {balance.map((entry) => {
              const total = entry.positive + entry.negative;
              const rate = total === 0 ? 0 : entry.positive / total;
              return (
                <div key={entry.split}>
                  <div className="flex items-baseline justify-between text-sm">
                    <span className="font-medium">{entry.split}</span>
                    <span className="text-muted-foreground">
                      {entry.positive.toLocaleString()} active · {entry.negative.toLocaleString()}{" "}
                      inactive ·{" "}
                      <span className={rate < 0.1 || rate > 0.9 ? "text-warning" : undefined}>
                        {percent(rate)} active
                      </span>
                    </span>
                  </div>
                  <div className="mt-1 flex h-2 gap-0.5 overflow-hidden rounded-full">
                    <div
                      className="bg-[var(--chart-1)]"
                      style={{ flex: entry.positive || 0.001 }}
                    />
                    <div className="bg-muted" style={{ flex: entry.negative || 0.001 }} />
                  </div>
                </div>
              );
            })}
            <p className="text-xs text-muted-foreground">
              Bars are active (coloured) against inactive. A partition under 10% active is flagged:
              MCC and AUPRC both get unstable there, and the number that comes back will move a lot
              between seeds.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Is this split a real generalization test?
 *
 * The a-priori version of the Scorecard's applicability number, asked from the
 * Dataset alone so it can be answered before any compute is spent.
 */
function SplitHonestySection({
  profile,
  isScaffoldSplit,
}: {
  profile: DatasetProfile;
  isScaffoldSplit: boolean;
}) {
  const similarity = profile.similarity;
  const scaffolds = profile.scaffolds;
  const leaked = scaffolds.cross_split_scaffolds > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Is this split a real test?</CardTitle>
        <p className="text-sm text-muted-foreground">
          How far each test compound sits from the nearest thing the model will train on. This is
          the applicability question asked before training rather than after it — a test set that is
          close to the training set will produce a flattering score no matter which engine runs.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-x-8 gap-y-4">
          {similarity ? (
            <>
              <Stat
                label="Median similarity"
                value={similarity.median.toFixed(2)}
                detail="Tanimoto over ECFP4, from each test compound to its nearest training compound."
              />
              <Stat
                label="Within domain"
                value={percent(similarity.within_domain)}
                detail={`Test compounds within ${similarity.within_domain_threshold} Tanimoto of the training set. The rest is extrapolation.`}
              />
              <Stat
                label="Near-duplicates"
                value={similarity.near_duplicates.toLocaleString()}
                tone={similarity.near_duplicates > 0 ? "warning" : undefined}
                detail={
                  similarity.near_duplicates > 0
                    ? `Test compounds at or above ${similarity.near_duplicate_threshold} Tanimoto to something in training. The model has effectively already seen these, and every metric is flattered by them.`
                    : `No test compound is within ${similarity.near_duplicate_threshold} Tanimoto of a training compound.`
                }
              />
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              Not measurable — this dataset has no train or no test partition.
            </p>
          )}
          <Stat
            label="Scaffolds on both sides"
            value={scaffolds.cross_split_scaffolds.toLocaleString()}
            tone={isScaffoldSplit && leaked ? "warning" : undefined}
            detail={
              isScaffoldSplit
                ? leaked
                  ? `${scaffolds.cross_split_compounds.toLocaleString()} compounds share a scaffold across train and test, which a scaffold split is supposed to prevent.`
                  : "Zero, which is what a scaffold split promises. Train and test share no ring system."
                : `${scaffolds.cross_split_compounds.toLocaleString()} compounds share a scaffold across train and test. Expected for a random split — and exactly what makes its scores optimistic.`
            }
          />
        </div>

        {similarity && (
          <HistogramChart
            bins={similarity.histogram}
            xLabel="nearest-neighbour Tanimoto to the training set"
            reference={{
              at: similarity.within_domain_threshold,
              label: "domain edge",
            }}
            caption="Mass piled to the right means the test set looks like the training set, and the benchmark is easier than it appears. Mass to the left means genuine extrapolation — a lower score there is worth more than a higher one on the right."
          />
        )}
      </CardContent>
    </Card>
  );
}

/** What chemistry is in here — congeneric series or diverse deck? */
function ScaffoldSection({ profile }: { profile: DatasetProfile }) {
  const scaffolds = profile.scaffolds;
  if (scaffolds.unique_count === 0) return null;

  const singletonShare = scaffolds.singleton_count / Math.max(scaffolds.unique_count, 1);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Chemical diversity</CardTitle>
        <p className="text-sm text-muted-foreground">
          Bemis-Murcko scaffolds. Hundreds of analogues of one core is a congeneric series: a model
          will interpolate across it beautifully and generalize nowhere. A long tail of singletons
          is a diverse deck, which is harder to fit and worth more when fitted.
        </p>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap gap-x-8 gap-y-4">
          <Stat
            label="Unique scaffolds"
            value={scaffolds.unique_count.toLocaleString()}
            detail={`${(profile.compounds / scaffolds.unique_count).toFixed(1)} compounds per scaffold on average.`}
          />
          <Stat
            label="Largest family"
            value={percent(scaffolds.largest_fraction)}
            detail="Share of the dataset sitting on its single most common scaffold."
          />
          <Stat
            label="One-off scaffolds"
            value={percent(singletonShare)}
            detail="Scaffolds represented by exactly one compound. A high share means little for a model to generalize from within any family."
          />
        </div>

        <CoverageCurveChart
          coverage={scaffolds.cumulative_coverage}
          caption="A curve that jumps to the top-left is a congeneric series; one that climbs gradually is a diverse deck."
        />

        {scaffolds.top.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Most common scaffolds
            </p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {scaffolds.top.map((entry) => (
                <div
                  key={entry.smiles || "acyclic"}
                  className="flex h-full flex-col items-center gap-2 rounded-lg border border-border p-3"
                >
                  {entry.smiles ? (
                    <StructureThumbnail smiles={entry.smiles} size={110} />
                  ) : (
                    <div className="flex h-[110px] w-[110px] items-center justify-center text-center text-xs text-muted-foreground">
                      no ring system
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground">
                    {entry.count.toLocaleString()} compound{entry.count === 1 ? "" : "s"} ·{" "}
                    {percent(entry.count / profile.compounds)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Would something trivial have done just as well? */
function DescriptorSection({ profile }: { profile: DatasetProfile }) {
  const descriptors = profile.descriptors;
  if (descriptors.length === 0) return null;

  const correlated = descriptors
    .filter((d) => d.target_correlation != null)
    .map((d) => ({
      label: descriptorLabel(d.name),
      value: d.target_correlation as number,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const strongest = correlated[0];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Physicochemical space</CardTitle>
        <p className="text-sm text-muted-foreground">
          Where this dataset sits in property space, and whether one ordinary descriptor already
          explains the target.
        </p>
      </CardHeader>
      <CardContent className="space-y-5">
        {strongest && (
          <div className="rounded-lg border border-border bg-muted/30 p-4">
            <p className="text-sm">
              <span className="font-medium">{strongest.label}</span> alone has a Spearman
              correlation of{" "}
              <span className="font-semibold tabular-nums">{strongest.value.toFixed(2)}</span> with
              the target.
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              {Math.abs(strongest.value) >= 0.7
                ? "That is most of the signal. A model that beats the baseline here has not yet shown it learned any chemistry beyond this one property — check it against the parity plot on the trained protocol before believing the headline."
                : Math.abs(strongest.value) >= 0.4
                  ? "A real but partial trend. Some of any model's score on this dataset is this property rather than chemistry."
                  : "Weak, which is good news: no single ordinary property explains this target, so a model has something genuine to learn."}
            </p>
          </div>
        )}

        {correlated.length > 0 && (
          <HorizontalBarChart
            data={correlated}
            xLabel="Spearman correlation with the target"
            diverging
            height={Math.max(160, correlated.length * 26 + 50)}
            caption="Sign is direction, length is strength. Descriptors that correlate negatively are exactly as informative as ones that correlate positively."
          />
        )}

        <div className="grid gap-5 sm:grid-cols-2">
          {descriptors.map((descriptor) => (
            <div key={descriptor.name}>
              <p className="text-xs font-medium">
                {descriptorLabel(descriptor.name)}
                <span className="ml-2 font-normal text-muted-foreground">
                  median {descriptor.median.toFixed(descriptor.median < 10 ? 2 : 0)}
                </span>
              </p>
              <SplitHistogramChart bins={descriptor.histogram} xLabel="" height={130} />
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Train and test are drawn as shares of their own partitions, so the shapes stay comparable
          despite the partitions being very different sizes. Two distributions that barely overlap
          are a covariate shift the split introduced.
        </p>
      </CardContent>
    </Card>
  );
}

/** The pairs no featurization can separate. */
function CliffSection({
  dataset,
  profile,
  unit,
}: {
  dataset: Dataset;
  profile: DatasetProfile;
  unit?: string | null;
}) {
  const cliffs = profile.activity_cliffs;
  const noiseFloor = dataset.validation_report.duplicate_spread;

  if (cliffs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Activity cliffs</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No near-identical pair disagrees about the target
            {profile.cliffs_sampled_from
              ? ` among the sample scanned of ${profile.cliffs_sampled_from.toLocaleString()} compounds`
              : ""}
            . Nothing here puts an extra ceiling on what a model can achieve.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Activity cliffs</CardTitle>
        <p className="text-sm text-muted-foreground">
          Near-identical structures the assay disagrees about. Any featurization that maps these two
          molecules to nearly the same point cannot predict both — collectively they are a second
          ceiling on accuracy, alongside the assay noise floor.
          {profile.cliffs_sampled_from
            ? ` Scanned on a sample of ${profile.cliffs_sampled_from.toLocaleString()} compounds.`
            : ""}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {cliffs.map((cliff) => (
          <div
            key={`${cliff.left_structure}-${cliff.right_structure}`}
            className="flex flex-wrap items-center gap-4 rounded-lg border border-border p-3"
          >
            <div className="flex flex-col items-center gap-1">
              <StructureThumbnail smiles={cliff.left_structure} size={96} />
              <ReadoutValue
                value={cliff.left_value}
                unit={unit}
                precision={2}
                className="text-xs"
              />
            </div>
            <div className="flex flex-col items-center gap-1">
              <StructureThumbnail smiles={cliff.right_structure} size={96} />
              <ReadoutValue
                value={cliff.right_value}
                unit={unit}
                precision={2}
                className="text-xs"
              />
            </div>
            <dl className="w-full max-w-[16rem] space-y-0.5 text-xs sm:w-auto sm:min-w-[12rem]">
              <div className="flex justify-between gap-2">
                <dt className="text-muted-foreground">similarity</dt>
                <dd className="tabular-nums">{cliff.similarity.toFixed(2)}</dd>
              </div>
              <div className="flex justify-between gap-2 border-t pt-0.5">
                <dt className="text-muted-foreground">differ by</dt>
                <dd className="font-medium text-warning">
                  <ReadoutValue value={cliff.delta} unit={unit} precision={2} />
                </dd>
              </div>
              {noiseFloor != null && (
                <div className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">assay noise</dt>
                  <dd className="tabular-nums">
                    <ReadoutValue value={noiseFloor} precision={2} />
                  </dd>
                </div>
              )}
            </dl>
          </div>
        ))}
        {noiseFloor != null && (
          <p className="text-xs text-muted-foreground">
            A gap far larger than the assay noise floor is a real structure-activity relationship. A
            gap close to it is two measurements that disagree, which is a data question rather than
            a chemistry one.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
