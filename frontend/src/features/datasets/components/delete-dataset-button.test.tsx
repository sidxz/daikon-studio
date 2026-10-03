import type { DatasetResponse } from "@/shared/lib/api/model";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeleteDatasetButton } from "./delete-dataset-button";

const hoisted = vi.hoisted(() => ({
  push: vi.fn(),
  mutate: vi.fn(),
  protocols: [] as { id: string; name: string; status: string; protocol_version: number }[],
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: hoisted.push }) }));
vi.mock("../hooks/use-datasets", () => ({
  useDeleteDataset: () => ({
    mutate: hoisted.mutate,
    reset: vi.fn(),
    isPending: false,
    error: null,
  }),
  useDatasetProtocols: () => ({ isLoading: false, data: { items: hoisted.protocols } }),
}));

const DATASET = { id: "d-1", name: "ben-inhibition", can_delete: true } as DatasetResponse;

describe("DeleteDatasetButton", () => {
  beforeEach(() => {
    hoisted.push.mockReset();
    hoisted.mutate.mockReset();
    hoisted.protocols = [];
  });

  it("deletes an unused dataset, then returns to the datasets list", () => {
    hoisted.mutate.mockImplementation((_id: string, options: { onSuccess: () => void }) =>
      options.onSuccess(),
    );
    render(<DeleteDatasetButton dataset={DATASET} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));

    expect(hoisted.mutate).toHaveBeenCalledWith("d-1", expect.anything());
    expect(hoisted.push).toHaveBeenCalledWith("/datasets");
  });

  it("lists the protocols trained on it and will not delete until they are gone", () => {
    hoisted.protocols = [
      { id: "p-1", name: "solubility rf", status: "draft", protocol_version: 1 },
      { id: "p-2", name: "solubility gp", status: "published", protocol_version: 2 },
    ];
    render(<DeleteDatasetButton dataset={DATASET} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(screen.getByText(/must be deleted first/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "solubility rf" })).toHaveAttribute(
      "href",
      "/protocols/p-1",
    );
    expect(screen.getByRole("button", { name: "Delete permanently" })).toBeDisabled();
  });
});
