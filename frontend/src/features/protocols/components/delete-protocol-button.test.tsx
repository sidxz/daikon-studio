import type { ProtocolResponse } from "@/shared/lib/api/model";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeleteProtocolButton } from "./delete-protocol-button";

const hoisted = vi.hoisted(() => ({
  push: vi.fn(),
  mutate: vi.fn(),
  error: null as Error | null,
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: hoisted.push }) }));
vi.mock("../hooks/use-protocols", () => ({
  useDeleteProtocol: () => ({
    mutate: hoisted.mutate,
    reset: vi.fn(),
    isPending: false,
    error: hoisted.error,
  }),
}));

const PROTOCOL = { id: "p-1", name: "solubility rf", can_delete: true } as ProtocolResponse;

describe("DeleteProtocolButton", () => {
  beforeEach(() => {
    hoisted.push.mockReset();
    hoisted.mutate.mockReset();
    hoisted.error = null;
  });

  it("deletes on confirmation, then returns to the protocols list", () => {
    hoisted.mutate.mockImplementation((_id: string, options: { onSuccess: () => void }) =>
      options.onSuccess(),
    );
    render(<DeleteProtocolButton protocol={PROTOCOL} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(screen.getByText(/the training run that produced it/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));

    expect(hoisted.mutate).toHaveBeenCalledWith("p-1", expect.anything());
    expect(hoisted.push).toHaveBeenCalledWith("/protocols");
  });

  it("shows the server's refusal inside the dialog", () => {
    hoisted.error = new Error("Published protocols cannot be deleted.");
    render(<DeleteProtocolButton protocol={PROTOCOL} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Published protocols cannot be deleted.");
  });
});
