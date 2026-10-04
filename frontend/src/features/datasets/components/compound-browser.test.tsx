import type { DatasetResponse } from "@/shared/lib/api/model";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CompoundBrowser } from "./compound-browser";

const hoisted = vi.hoisted(() => ({ queries: [] as Record<string, unknown>[] }));

vi.mock("@/shared/components/chemistry/structure-thumbnail", () => ({
  StructureThumbnail: () => null,
}));
vi.mock("../hooks/use-datasets", () => ({
  useDatasetCompounds: (_id: string, query: Record<string, unknown>) => {
    hoisted.queries.push(query);
    return {
      isLoading: false,
      isError: false,
      data: {
        items: [
          { structure: "CCO", targets: { y: 1, active: 0 }, split: "train", compound_id: "RU-7" },
        ],
        total: 1,
      },
    };
  },
}));

const dataset = (idColumn: string | null) =>
  ({
    id: "d-1",
    id_column: idColumn,
    targets: [
      { column: "y", kind: "numeric", unit: null },
      { column: "active", kind: "binary", unit: null },
    ],
  }) as unknown as DatasetResponse;

describe("CompoundBrowser", () => {
  beforeEach(() => {
    hoisted.queries = [];
  });

  it("shows each compound's ID and searches by it", async () => {
    vi.useFakeTimers();
    render(<CompoundBrowser dataset={dataset("RU ID")} />);

    expect(screen.getByText("RU-7")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("searchbox", { name: "Search by ID" }), {
      target: { value: "RU-7" },
    });
    await act(async () => vi.advanceTimersByTime(400));

    expect(hoisted.queries.at(-1)).toMatchObject({ q: "RU-7", offset: 0 });
    vi.useRealTimers();
  });

  it("shows a column per target and sorts by the one clicked", () => {
    render(<CompoundBrowser dataset={dataset("RU ID")} />);
    expect(screen.getByRole("columnheader", { name: "active" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /active/ }));
    expect(hoisted.queries.at(-1)).toMatchObject({ sort: "target", target: 1 });
  });

  it("offers no search without an identifier column", () => {
    render(<CompoundBrowser dataset={dataset(null)} />);
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
  });
});
