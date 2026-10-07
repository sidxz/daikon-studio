"use client";

import { EpochCurveChart, type EpochLine } from "@/shared/components/charts/charts";
import { ChartLegend } from "@/shared/components/charts/plot-figure";
import { type ChartTheme, useChartTheme } from "@/shared/components/charts/use-chart-theme";
import { Badge } from "@/shared/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Progress } from "@/shared/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import type { EpochResponse } from "@/shared/lib/api/model";
import { useMemo, useState } from "react";
import {
  type EpochSeries,
  SCORE_LABELS,
  durationLabel,
  groupSeries,
  higherIsBetter,
  isFinished,
  keptEpoch,
  keptRule,
  scoreNames,
  secondsLeft,
  secondsPerEpoch,
} from "../lib/training-epochs";

const DASHES = [undefined, "6,4", "2,3"];

function format(name: string, value: number): string {
  return name === "rmse" || name === "mae" ? value.toPrecision(3) : value.toFixed(3);
}

function lineOf(
  series: EpochSeries,
  label: string,
  color: string,
  dash: string | undefined,
  pick: (point: EpochResponse) => number | null | undefined,
): EpochLine {
  return {
    label,
    color,
    dash,
    values: series.points
      .map((point) => ({ epoch: point.epoch, value: pick(point) }))
      .filter((point): point is { epoch: number; value: number } => point.value != null),
  };
}

/** The best value a score reached, and when, honoring which direction is better. */
function bestOf(series: EpochSeries, name: string): EpochResponse | null {
  let best: EpochResponse | null = null;
  for (const point of series.points) {
    const value = point.scores[name];
    if (value === undefined) continue;
    const current = best?.scores[name];
    if (current === undefined || (higherIsBetter(name) ? value > current : value < current)) {
      best = point;
    }
  }
  return best;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
      {hint && <p className="truncate text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/** The charts and numbers for one model: what it is doing now, epoch by epoch. */
function SeriesView({
  series,
  theme,
  live,
}: {
  series: EpochSeries;
  theme: ChartTheme;
  live: boolean;
}) {
  const last = series.points[series.points.length - 1];
  const kept = keptEpoch(series);
  const pace = secondsPerEpoch(series);
  const left = secondsLeft(series);
  const names = scoreNames(series.points);
  const errors = names.filter((name) => name === "rmse" || name === "mae");
  const ratios = names.filter((name) => name !== "rmse" && name !== "mae");

  const losses = [
    lineOf(series, "training", theme.pair[0], undefined, (point) => point.train_loss),
    lineOf(series, "validation", theme.pair[1], DASHES[1], (point) => point.val_loss),
  ].filter((line) => line.values.length > 0);
  const scoreLines = (group: string[]) =>
    group.map((name, index) =>
      lineOf(
        series,
        SCORE_LABELS[name],
        theme.triple[index % 3],
        DASHES[index % 3],
        (point) => point.scores[name],
      ),
    );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Epoch" value={`${last.epoch} of ${last.epochs}`} />
        <Stat label="Per epoch" value={pace === null ? "N/A" : durationLabel(pace)} />
        <Stat
          label="Left for this model"
          value={left === null ? "N/A" : left === 0 ? "done" : `about ${durationLabel(left)}`}
        />
        <Stat
          label={live ? "Kept so far" : "Kept epoch"}
          value={kept ? `epoch ${kept.epoch}` : "N/A"}
          hint={kept ? keptRule(series) : "no validation set"}
        />
      </div>

      {names.length > 0 && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {names.map((name) => {
            const best = bestOf(series, name);
            const value = last.scores[name];
            return (
              <Stat
                key={name}
                label={`Validation ${SCORE_LABELS[name]}${name === "mcc" ? " (0.5 cutoff)" : ""}`}
                value={value === undefined ? "N/A" : format(name, value)}
                hint={
                  best && best !== last
                    ? `best ${format(name, best.scores[name])} at epoch ${best.epoch}`
                    : undefined
                }
              />
            );
          })}
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {losses.length > 0 && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <p className="text-sm font-medium">Loss</p>
              <ChartLegend items={losses.map(({ label, color }) => ({ label, color }))} />
            </div>
            <EpochCurveChart
              lines={losses}
              epochs={last.epochs}
              kept={kept?.epoch}
              yLabel="loss"
              caption="Both should fall at first. When validation loss turns upward while training loss keeps falling, the model fits its training compounds better than new ones, often by growing overconfident. Its ranking of new compounds can still improve, as the scores on the right show."
            />
          </div>
        )}
        {[ratios, errors]
          .filter((group) => group.length > 0)
          .map((group) => {
            const lines = scoreLines(group);
            return (
              <div key={group.join()} className="space-y-2">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-sm font-medium">
                    {group === errors
                      ? "Validation error, in the target's unit"
                      : "Validation scores"}
                  </p>
                  <ChartLegend items={lines.map(({ label, color }) => ({ label, color }))} />
                </div>
                <EpochCurveChart
                  lines={lines}
                  epochs={last.epochs}
                  kept={kept?.epoch}
                  yLabel={group === errors ? "error" : "score"}
                  format={(value) => (group === errors ? value.toPrecision(3) : value.toFixed(3))}
                  caption={
                    group === errors
                      ? "Lower is better."
                      : "Measured on the validation compounds after each epoch; higher is better."
                  }
                />
              </div>
            );
          })}
      </div>
    </div>
  );
}

/** Every model the run has trained, as a table: the readable form of the charts. */
function SeriesTable({ series, names }: { series: EpochSeries[]; names: string[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead className="text-right">Epochs</TableHead>
          <TableHead className="text-right">Kept epoch</TableHead>
          <TableHead className="text-right">Validation loss</TableHead>
          {names.map((name) => (
            <TableHead key={name} className="text-right">
              {SCORE_LABELS[name]}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {series.map((one) => {
          const last = one.points[one.points.length - 1];
          const kept = keptEpoch(one) ?? last;
          return (
            <TableRow key={one.key}>
              <TableCell>{one.label}</TableCell>
              <TableCell className="text-right tabular-nums">
                {last.epoch} of {last.epochs}
              </TableCell>
              <TableCell className="text-right tabular-nums">{kept.epoch}</TableCell>
              <TableCell className="text-right tabular-nums">
                {kept.val_loss == null ? "N/A" : kept.val_loss.toFixed(3)}
              </TableCell>
              {names.map((name) => (
                <TableCell key={name} className="text-right tabular-nums">
                  {kept.scores[name] === undefined ? "N/A" : format(name, kept.scores[name])}
                </TableCell>
              ))}
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

/**
 * A training run's epochs, live or after the fact. Live, the model training now leads
 * and the ones already finished follow as a table; afterwards, any model can be
 * charted. Engines that train in epochs (chemprop, MoLFormer) report them; for the
 * others there are none and the run page shows only its progress bar.
 */
export function TrainingProgress({
  points,
  live,
  phase,
  progress,
}: {
  points: EpochResponse[];
  live: boolean;
  phase?: string | null;
  progress?: number;
}) {
  const theme = useChartTheme();
  const series = useMemo(() => groupSeries(points), [points]);
  const names = useMemo(() => scoreNames(points), [points]);
  const [chosen, setChosen] = useState<string | null>(null);
  const newest = series[series.length - 1];
  // Live, the newest model is the one training only until its last epoch. After that the
  // run has moved to a stage that reports no epochs (a tree baseline such as XGBoost), so
  // the finished models are charted as they are after the run, with a choice of model.
  const trainingNow = live && newest !== undefined && !isFinished(newest);
  const current = trainingNow ? newest : (series.find((one) => one.key === chosen) ?? newest);
  const finished = trainingNow ? series.slice(0, -1) : series;

  return (
    <Card>
      {!live && (
        <CardHeader>
          <CardTitle className="text-base">How training went</CardTitle>
          <p className="text-sm text-muted-foreground">
            Each model, epoch by epoch, as it trained. Scores are on the validation compounds.
          </p>
        </CardHeader>
      )}
      <CardContent className="space-y-6 py-6">
        {live && (
          <div className="space-y-2">
            <div className="flex items-baseline justify-between gap-4 text-sm">
              <span className="text-muted-foreground">{phase ?? "Starting…"}</span>
              {progress !== undefined && (
                <span className="tabular-nums text-muted-foreground">
                  {Math.round(progress * 100)}% of the run
                </span>
              )}
            </div>
            <Progress value={Math.round((progress ?? 0) * 100)} />
          </div>
        )}

        {live && !trainingNow && series.length > 0 && (
          <p className="text-sm text-muted-foreground">
            The stage training now does not report epochs, so it has no charts. These are the models
            that finished.
          </p>
        )}

        {current && theme && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              {trainingNow || series.length < 2 ? (
                <p className="text-sm font-medium">
                  {trainingNow ? "Training now: " : ""}
                  {current.label}
                </p>
              ) : (
                <Select value={current.key} onValueChange={setChosen}>
                  <SelectTrigger className="w-auto min-w-64" aria-label="Model to chart">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {series.map((one) => (
                      <SelectItem key={one.key} value={one.key}>
                        {one.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {current.points[0]?.device && (
                <Badge variant="outline" className="font-mono text-xs">
                  {current.points[0].device}
                </Badge>
              )}
            </div>
            <SeriesView series={current} theme={theme} live={trainingNow} />
          </div>
        )}

        {finished.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">{live ? "Finished models" : "Every model"}</p>
            <SeriesTable series={finished} names={names} />
            <p className="text-xs text-muted-foreground">
              Validation loss and scores at each model's kept epoch.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
