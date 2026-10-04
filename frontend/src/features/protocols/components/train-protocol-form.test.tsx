import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TrainProtocolForm, comparesAgainstItself, resolveConditions } from "./train-protocol-form";

const SPECS = [{ key: "n_estimators", label: "Trees", type: "integer", default: 500, options: [] }];

describe("comparesAgainstItself", () => {
  it("is true when both sides fall back to the same defaults", () => {
    expect(comparesAgainstItself("rf", {}, "rf", {}, SPECS, SPECS)).toBe(true);
  });

  it("is true when one side sets a value the other takes as its default", () => {
    // The trap: comparing raw form state would call these different, and the
    // warning would go missing on a run the server treats as a self-comparison.
    expect(comparesAgainstItself("rf", { n_estimators: 500 }, "rf", {}, SPECS, SPECS)).toBe(true);
  });

  it("is false for the same engine with genuinely different settings", () => {
    expect(comparesAgainstItself("rf", { n_estimators: 100 }, "rf", {}, SPECS, SPECS)).toBe(false);
  });

  it("is false for different engines", () => {
    expect(comparesAgainstItself("rf", {}, "xgb", {}, SPECS, SPECS)).toBe(false);
  });
});

// --- CheMeleon pinning: the submitted payload must carry the pinned values ---
//
// Guards against a real regression: a future edit to `submit()` that posts
// `resolveConditions(specs, conditions)` instead of raw `conditions` would
// pass every other test in this suite while silently dropping the merged
// pin, because `resolveConditions` only fills in *manifest* defaults and has
// no idea `depth`/`message_hidden_dim` were just overwritten by
// `PINNED_BY_PRETRAINED`. That would resurrect the exact bug this task
// exists to prevent: a Protocol recording `depth: 3` for a fit that actually
// ran at 6.

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

const CHEMPROP_ENGINE = {
  id: "chemprop-dmpnn",
  version: "1.0.0",
  name: "Chemprop D-MPNN",
  description: "",
  tasks: ["regression", "binary_classification"],
  supports_multitask: true,
  is_baseline: false,
  conditions: [
    {
      key: "depth",
      label: "Message passing steps",
      type: "integer",
      default: 3,
      minimum: 2,
      maximum: 6,
      required: false,
      options: [],
    },
    {
      key: "message_hidden_dim",
      label: "Hidden size",
      type: "integer",
      default: 300,
      minimum: 64,
      maximum: 2400,
      required: false,
      options: [],
    },
    {
      key: "pretrained",
      label: "Pretrained weights",
      type: "enum",
      default: "none",
      options: ["none", "CheMeleon"],
      required: false,
    },
  ],
};
const RF_ENGINE = {
  id: "ecfp4-rf",
  version: "1.0.0",
  name: "ECFP4 + RF",
  description: "",
  tasks: ["regression", "binary_classification"],
  supports_multitask: false,
  is_baseline: true,
  conditions: [],
};

// An engine with a classification-only setting, to see it stripped from a numeric dataset.
const WEIGHTED_ENGINE = {
  id: "weighted-forest",
  version: "1.0.0",
  name: "Weighted forest",
  description: "",
  tasks: ["regression", "binary_classification"],
  supports_multitask: false,
  is_baseline: false,
  conditions: [
    {
      key: "positive_weighting",
      label: "Positive-class weighting",
      type: "enum",
      default: "none",
      options: ["none", "balanced"],
      option_labels: ["None", "Balanced"],
      required: false,
      tasks: ["binary_classification"],
    },
  ],
};

// One numeric target unless a test says otherwise; the mixed-kind case is the
// one that withholds a joint engine.
const hoisted = vi.hoisted(() => ({
  targets: [{ kind: "numeric", column: "logS" }] as { kind: string; column: string }[],
}));

vi.mock("@/features/datasets", () => ({
  useDatasets: () => ({
    isLoading: false,
    data: { items: [{ id: "ds-1", name: "Solubility", row_count: 100 }] },
  }),
  useDataset: () => ({
    data: {
      id: "ds-1",
      targets: hoisted.targets,
      split: { strategy: "scaffold" },
    },
  }),
}));

const mutateAsync = vi.fn().mockResolvedValue({ id: "run-1", status: "queued" });

vi.mock("../hooks/use-protocols", async () => {
  const actual =
    await vi.importActual<typeof import("../hooks/use-protocols")>("../hooks/use-protocols");
  return {
    ...actual,
    useTrainProtocol: () => ({ mutateAsync, isPending: false }),
    useRunPoll: () => ({ data: undefined }),
  };
});

vi.mock("@/features/engines", async () => {
  const actual = await vi.importActual<typeof import("@/features/engines")>("@/features/engines");
  return {
    ...actual,
    useEngines: () => ({ data: [CHEMPROP_ENGINE, RF_ENGINE, WEIGHTED_ENGINE] }),
  };
});

describe("engines for several targets", () => {
  afterEach(() => {
    hoisted.targets = [{ kind: "numeric", column: "logS" }];
  });

  it("offers only engines that can train every target, and says why a joint one is missing", async () => {
    hoisted.targets = [
      { kind: "numeric", column: "logS" },
      { kind: "binary", column: "reactive" },
    ];
    render(
      <QueryClientProvider client={new QueryClient()}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));

    expect(
      await screen.findByText(
        /Chemprop D-MPNN trains one joint model and needs targets of a single kind, so it is not offered/,
      ),
    ).toBeInTheDocument();
    // The joint engine is not offered for a mixed-kind dataset.
    fireEvent.click(screen.getByText("Choose an engine"));
    expect(await screen.findByRole("option", { name: /ECFP4 \+ RF/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /Chemprop/ })).not.toBeInTheDocument();
  });
});

describe("CheMeleon's pinned settings reach the submitted payload", () => {
  afterEach(() => mutateAsync.mockClear());

  it("posts message_hidden_dim=2048 and depth=6, not the manifest defaults", async () => {
    const qc = new QueryClient();
    render(
      <QueryClientProvider client={qc}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));

    fireEvent.click(screen.getByText("Choose an engine"));
    fireEvent.click(await screen.findByText("Chemprop D-MPNN"));

    // Pick CheMeleon in the chosen-engine settings block.
    const pretrainedTrigger = await screen.findByText("none");
    fireEvent.click(pretrainedTrigger);
    fireEvent.click(await screen.findByText("CheMeleon"));

    // Displayed values are pinned and disabled -- the same fact the merge
    // effect must also have written into submitted state.
    await waitFor(() => {
      const hidden = screen.getByLabelText(/Hidden size/) as HTMLInputElement;
      const depth = screen.getByLabelText(/Message passing steps/) as HTMLInputElement;
      expect(hidden.value).toBe("2048");
      expect(depth.value).toBe("6");
      expect(hidden).toBeDisabled();
      expect(depth).toBeDisabled();
    });

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol" } });
    fireEvent.click(screen.getByText("Train"));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    const payload = mutateAsync.mock.calls[0][0];
    expect(payload.conditions.message_hidden_dim).toBe(2048);
    expect(payload.conditions.depth).toBe(6);
  });
});

// --- Submit is blocked when a run would silently compare an engine against
// itself, exempting only the registry's flagged baseline engine. ---
//
// Before this branch that state was reachable only by the flagged baseline
// engine, for which "there is nothing to compare against" is true. With a
// choosable baseline, any engine can now be set to compare against itself; the
// server still runs one fit and reports it as both sides, so the client must
// refuse to submit rather than let the Scorecard render a comparison that
// never happened.

describe("submit is blocked when the run would compare an engine against itself", () => {
  afterEach(() => mutateAsync.mockClear());

  it("disables Train for a non-baseline engine compared against itself", async () => {
    const qc = new QueryClient();
    render(
      <QueryClientProvider client={qc}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));

    // Indexed rather than by trigger text: once both selects have shown
    // "Chemprop D-MPNN" at some point, the text alone is no longer unique.
    const [, engineTrigger, baselineTrigger] = screen.getAllByRole("combobox");

    fireEvent.click(engineTrigger);
    fireEvent.click(await screen.findByRole("option", { name: "Chemprop D-MPNN" }));

    // The baseline defaults to the flagged engine (ECFP4 + RF); switch it to
    // chemprop too, so both sides are chemprop-dmpnn on identical settings.
    fireEvent.click(baselineTrigger);
    fireEvent.click(await screen.findByRole("option", { name: "Chemprop D-MPNN" }));

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol" } });

    expect(screen.getByText("Train")).toBeDisabled();
    fireEvent.click(screen.getByText("Train"));
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it("still permits Train when the flagged baseline engine compares against itself", async () => {
    const qc = new QueryClient();
    render(
      <QueryClientProvider client={qc}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));

    // The baseline already defaults to this same flagged engine, with no
    // conditions on either side -- a self-comparison this product has always
    // allowed, because there is nothing else to compare the baseline against.
    fireEvent.click(screen.getByText("Choose an engine"));
    fireEvent.click(await screen.findByRole("option", { name: /ECFP4 \+ RF/ }));

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol" } });

    await waitFor(() => expect(screen.getByText("Train")).not.toBeDisabled());
    fireEvent.click(screen.getByText("Train"));
    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
  });
});

// --- Tuning decision cutoffs is offered only where there is a cutoff to tune ---

describe("the tune-cutoffs option", () => {
  afterEach(() => {
    mutateAsync.mockClear();
    hoisted.targets = [{ kind: "numeric", column: "logS" }];
  });

  async function trainWithRf() {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));
    return async () => {
      fireEvent.click(screen.getByText("Choose an engine"));
      fireEvent.click(await screen.findByRole("option", { name: /ECFP4 \+ RF/ }));
      fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol" } });
      await waitFor(() => expect(screen.getByText("Train")).not.toBeDisabled());
      fireEvent.click(screen.getByText("Train"));
      await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
      return mutateAsync.mock.calls[0][0];
    };
  }

  it("is absent before a dataset is chosen", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );
    expect(screen.queryByRole("checkbox", { name: "Tune decision cutoffs" })).toBeNull();
  });

  it("sends tune_cutoffs: true when checked on a dataset with a binary target", async () => {
    hoisted.targets = [{ kind: "binary", column: "reactive" }];
    const submit = await trainWithRf();
    fireEvent.click(await screen.findByRole("checkbox", { name: "Tune decision cutoffs" }));
    expect((await submit()).tune_cutoffs).toBe(true);
  });

  it("sends tune_cutoffs: false by default", async () => {
    hoisted.targets = [{ kind: "binary", column: "reactive" }];
    const submit = await trainWithRf();
    await screen.findByRole("checkbox", { name: "Tune decision cutoffs" });
    expect((await submit()).tune_cutoffs).toBe(false);
  });

  it("is absent for a numeric-only dataset", async () => {
    const submit = await trainWithRf();
    expect(screen.queryByRole("checkbox", { name: "Tune decision cutoffs" })).toBeNull();
    expect((await submit()).tune_cutoffs).toBe(false);
  });
});

// --- A setting the form hides for this dataset is not submitted ---

describe("settings that do not apply to the dataset", () => {
  afterEach(() => {
    mutateAsync.mockClear();
    hoisted.targets = [{ kind: "numeric", column: "logS" }];
  });

  async function trainWeightedForest(targets: { kind: string; column: string }[]) {
    hoisted.targets = targets;
    render(
      <QueryClientProvider client={new QueryClient()}>
        <TrainProtocolForm />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByText(/Solubility/));
    fireEvent.click(screen.getByText("Choose an engine"));
    fireEvent.click(await screen.findByRole("option", { name: /Weighted forest/ }));
    // Set the weighting while it is visible, as a scientist who then changes dataset would.
    fireEvent.click(await screen.findByRole("combobox", { name: "Positive-class weighting" }));
    fireEvent.click(await screen.findByRole("option", { name: "Balanced" }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol" } });
  }

  it("drops a stale value once the dataset has no classification target", async () => {
    await trainWeightedForest([{ kind: "binary", column: "reactive" }]);
    hoisted.targets = [{ kind: "numeric", column: "logS" }];
    // Any state change re-renders the form against the new dataset.
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Test protocol 2" } });
    await waitFor(() => expect(screen.getByText("Train")).not.toBeDisabled());
    fireEvent.click(screen.getByText("Train"));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    expect(mutateAsync.mock.calls[0][0].conditions).toEqual({});
  });

  it("sends the value while the dataset has a classification target", async () => {
    await trainWeightedForest([{ kind: "binary", column: "reactive" }]);
    await waitFor(() => expect(screen.getByText("Train")).not.toBeDisabled());
    fireEvent.click(screen.getByText("Train"));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    expect(mutateAsync.mock.calls[0][0].conditions).toEqual({ positive_weighting: "balanced" });
  });
});
