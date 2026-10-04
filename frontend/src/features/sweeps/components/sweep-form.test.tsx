import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SweepForm } from "./sweep-form";

function Wrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

const RF_ENGINE = {
  id: "ecfp4-rf",
  version: "1.0.0",
  name: "ECFP4 + RF",
  description: "",
  tasks: ["regression", "binary_classification"],
  supports_multitask: false,
  is_baseline: true,
  conditions: [
    {
      key: "n_estimators",
      label: "Trees",
      type: "integer",
      default: 500,
      minimum: 10,
      maximum: 2000,
      required: false,
      options: [],
    },
  ],
};

const hoisted = vi.hoisted(() => ({
  targets: [{ kind: "numeric", column: "logS", unit: null }] as {
    kind: string;
    column: string;
    unit: null;
  }[],
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

vi.mock("@/features/engines", async () => {
  const actual = await vi.importActual<typeof import("@/features/engines")>("@/features/engines");
  return {
    ...actual,
    useEngines: () => ({ data: [RF_ENGINE] }),
  };
});

const mutateAsync = vi.fn().mockResolvedValue({ sweep_id: "sweep-1" });

vi.mock("../hooks/use-sweeps", () => ({
  useSubmitSweep: () => ({ mutateAsync, isPending: false }),
}));

describe("SweepForm", () => {
  it("starts with one config row", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(1);
  });

  it("adds and removes config rows", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    fireEvent.click(screen.getByRole("button", { name: /add configuration/i }));
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(2);
    fireEvent.click(screen.getAllByRole("button", { name: /remove configuration/i })[0]);
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(1);
  });

  it("will not remove the last config row", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    expect(screen.queryByRole("button", { name: /remove configuration/i })).toBeNull();
  });
});

// --- Submit resolves conditions against manifest defaults, not raw form state ---
//
// The subtlest thing this form does: a config row whose condition input was
// never touched must still submit `{n_estimators: 500}`, not `{}`. Raw form
// state would make those two payloads look different for a fit the server
// resolves identically -- defeating the cache key and making one fit look
// like two different sweep members.

describe("submit resolves conditions against manifest defaults", () => {
  afterEach(() => mutateAsync.mockClear());

  it("sends the manifest default for an untouched condition", async () => {
    render(<SweepForm />, { wrapper: Wrapper });

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByRole("option", { name: /Solubility/ }));

    fireEvent.click(screen.getByText("Choose an engine"));
    fireEvent.click(await screen.findByRole("option", { name: /ECFP4 \+ RF/ }));

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Sweep" } });
    fireEvent.click(screen.getByText("Start sweep"));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    const payload = mutateAsync.mock.calls[0][0];
    expect(payload.configs).toEqual([{ engine_id: "ecfp4-rf", conditions: { n_estimators: 500 } }]);
  });
});

// --- Submit is blocked while any config row has no engine picked ---
//
// A config row with `engine_id: ""` posts to a server that pre-flights
// nothing wrong with an empty string, creates no runs, and returns "Engine
// not found" with no indication of which of N rows was the empty one.

describe("submit requires an engine on every config row", () => {
  afterEach(() => mutateAsync.mockClear());

  it("disables submit while the only config row has no engine", async () => {
    render(<SweepForm />, { wrapper: Wrapper });

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByRole("option", { name: /Solubility/ }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Sweep" } });

    expect(screen.getByText("Start sweep")).toBeDisabled();
    fireEvent.click(screen.getByText("Start sweep"));
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it("re-disables submit when a newly added row has no engine yet", async () => {
    render(<SweepForm />, { wrapper: Wrapper });

    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByRole("option", { name: /Solubility/ }));
    fireEvent.click(screen.getByText("Choose an engine"));
    fireEvent.click(await screen.findByRole("option", { name: /ECFP4 \+ RF/ }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Sweep" } });

    await waitFor(() => expect(screen.getByText("Start sweep")).not.toBeDisabled());

    fireEvent.click(screen.getByRole("button", { name: /add configuration/i }));

    expect(screen.getByText("Start sweep")).toBeDisabled();
  });
});

// --- Tuning decision cutoffs is offered only where there is a cutoff to tune ---

describe("the tune-cutoffs option", () => {
  afterEach(() => {
    mutateAsync.mockClear();
    hoisted.targets = [{ kind: "numeric", column: "logS", unit: null }];
  });

  async function startSweep() {
    render(<SweepForm />, { wrapper: Wrapper });
    fireEvent.click(screen.getByText("Choose a dataset"));
    fireEvent.click(await screen.findByRole("option", { name: /Solubility/ }));
    return async () => {
      fireEvent.click(screen.getByText("Choose an engine"));
      fireEvent.click(await screen.findByRole("option", { name: /ECFP4 \+ RF/ }));
      fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Sweep" } });
      fireEvent.click(screen.getByText("Start sweep"));
      await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
      return mutateAsync.mock.calls[0][0];
    };
  }

  it("sends tune_cutoffs: true when checked on a dataset with a binary target", async () => {
    hoisted.targets = [{ kind: "binary", column: "reactive", unit: null }];
    const submit = await startSweep();
    fireEvent.click(await screen.findByRole("checkbox", { name: "Tune decision cutoffs" }));
    expect((await submit()).tune_cutoffs).toBe(true);
  });

  it("is absent, and sends false, for a numeric-only dataset", async () => {
    const submit = await startSweep();
    expect(screen.queryByRole("checkbox", { name: "Tune decision cutoffs" })).toBeNull();
    expect((await submit()).tune_cutoffs).toBe(false);
  });
});
