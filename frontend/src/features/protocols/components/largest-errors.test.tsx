import type { ScorecardResponse } from "@/shared/lib/api/model";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LargestErrors } from "./largest-errors";

// The real thumbnail draws with RDKit; the order of the SMILES is what matters here.
vi.mock("@/shared/components/chemistry/structure-thumbnail", () => ({
  StructureThumbnail: ({ smiles }: { smiles: string }) => <span data-testid="mol">{smiles}</span>,
}));

const row = (structure: string, scaffold: string, residual: number) => ({
  structure,
  scaffold,
  residual,
  actual: 0,
  predicted: residual,
  similarity: 0.5,
});

const scorecard = (rows: ReturnType<typeof row>[]) =>
  ({ worst_rows: rows, unit: null }) as unknown as ScorecardResponse;

const order = () => screen.getAllByTestId("mol").map((node) => node.textContent);

describe("LargestErrors", () => {
  it("groups a shared scaffold side by side, tags it, and reorders by error on request", () => {
    render(
      <LargestErrors
        scorecard={scorecard([
          row("a", "c1ccccc1", 0.5),
          row("b", "C1CC1", 0.9),
          row("c", "c1ccccc1", 0.7),
        ])}
      />,
    );

    expect(order()).toEqual(["c", "a", "b"]);
    expect(screen.getAllByText("Series A · 2")).toHaveLength(2);
    expect(screen.getByText(/Shared scaffolds: A \(2\)\./)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "By error" }));

    expect(order()).toEqual(["b", "c", "a"]);
    expect(screen.getByRole("button", { name: "By error" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("offers no toggle when no two compounds share a scaffold", () => {
    render(<LargestErrors scorecard={scorecard([row("a", "c1ccccc1", 0.5), row("b", "", 0.9)])} />);

    expect(order()).toEqual(["b", "a"]);
    expect(screen.queryByRole("button", { name: "By series" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Series A/)).not.toBeInTheDocument();
    expect(screen.getByText(/No two share a scaffold\./)).toBeInTheDocument();
  });
});
