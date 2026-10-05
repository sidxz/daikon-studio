import type { EpochResponse } from "@/shared/lib/api/model";

/** One epoch of a training run, a minute after the last, for the run-page tests. */
const start = Date.parse("2026-10-04T20:00:00Z");

export function epoch(n: number, overrides: Partial<EpochResponse> = {}): EpochResponse {
  return {
    fit: "model",
    target: null,
    member: null,
    members: null,
    epoch: n,
    epochs: 50,
    train_loss: 0.7 - n / 100,
    val_loss: 0.69 - n / 200,
    scores: { auroc: 0.7 + n / 100, auprc: 0.4, mcc: 0.2 },
    device: "cuda:0",
    kept_epoch: null,
    kept_by: null,
    at: new Date(start + n * 60_000).toISOString(), // a minute an epoch
    ...overrides,
  };
}
