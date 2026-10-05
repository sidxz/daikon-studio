import { describe, expect, it } from "vitest";
import {
  durationLabel,
  groupSeries,
  keptEpoch,
  keptRule,
  scoreNames,
  secondsLeft,
  secondsPerEpoch,
  seriesLabel,
} from "./training-epochs";
import { epoch } from "./training-epochs.fixture";

describe("groupSeries", () => {
  it("splits a run into one series per stage, target and ensemble model, in order", () => {
    const series = groupSeries([
      epoch(1, { target: "aggregator" }),
      epoch(1, { target: "reactive" }),
      epoch(2, { target: "aggregator" }),
      epoch(1, { fit: "random-split", target: "aggregator", member: 2, members: 3 }),
    ]);

    expect(series.map((one) => one.label)).toEqual([
      "aggregator",
      "reactive",
      "Random-split comparison · aggregator · model 2 of 3",
    ]);
    expect(series[0].points.map((point) => point.epoch)).toEqual([1, 2]);
  });

  it("names a single model fitting every target at once plainly", () => {
    expect(seriesLabel(epoch(1))).toBe("One model for all targets");
  });
});

describe("keptEpoch", () => {
  it("is the epoch the fit reports, whatever its loss", () => {
    const kept = { kept_epoch: 3, kept_by: "auprc" };
    const [series] = groupSeries([
      epoch(1, { val_loss: 0.5, kept_epoch: 1, kept_by: "auprc" }),
      epoch(2, { val_loss: 0.4, kept_epoch: 1, kept_by: "auprc" }),
      epoch(3, { val_loss: 0.45, ...kept }),
      epoch(4, { val_loss: 0.6, ...kept }),
    ]);
    expect(keptEpoch(series)?.epoch).toBe(3);
    expect(keptRule(series)).toBe("best validation PR AUC");
  });

  it("is the lowest validation loss for a fit recorded before fits reported it", () => {
    const [series] = groupSeries([
      epoch(1, { val_loss: 0.5 }),
      epoch(2, { val_loss: 0.4 }),
      epoch(3, { val_loss: 0.45 }),
    ]);
    expect(keptEpoch(series)?.epoch).toBe(2);
    expect(keptRule(series)).toBe("lowest validation loss");
  });

  it("is null for a fit with no validation set", () => {
    const [series] = groupSeries([epoch(1, { val_loss: null })]);
    expect(keptEpoch(series)).toBeNull();
  });
});

describe("pace", () => {
  it("reads seconds per epoch off the timestamps and projects the rest of the fit", () => {
    const [series] = groupSeries([epoch(1), epoch(2), epoch(3)]);
    expect(secondsPerEpoch(series)).toBe(60);
    expect(secondsLeft(series)).toBe(47 * 60);
  });

  it("says nothing from a single epoch", () => {
    const [series] = groupSeries([epoch(1)]);
    expect(secondsLeft(series)).toBeNull();
  });

  it("labels durations at planning precision", () => {
    expect(durationLabel(42)).toBe("42 s");
    expect(durationLabel(14 * 60)).toBe("14 min");
    expect(durationLabel(125 * 60)).toBe("2 h 5 min");
  });
});

describe("scoreNames", () => {
  it("lists the scores present, in the app's vocabulary order", () => {
    expect(scoreNames([epoch(1, { scores: { mcc: 0.1, auroc: 0.8 } })])).toEqual(["auroc", "mcc"]);
    expect(scoreNames([epoch(1, { scores: { r2: 0.5, rmse: 1.2 } })])).toEqual(["rmse", "r2"]);
  });
});
