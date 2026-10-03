import type { DatasetResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { IdColumnField } from "./id-column-field";

vi.mock("../hooks/use-datasets", () => ({
  useDatasetColumns: () => ({ data: { columns: ["name", "num"] } }),
  useSetDatasetIdColumn: () => ({ mutate: vi.fn(), isPending: false }),
}));

const dataset = (overrides: Partial<DatasetResponse>) =>
  ({ id: "d-1", id_column: "name", can_edit: true, ...overrides }) as DatasetResponse;

describe("IdColumnField", () => {
  it("lets an editor change it", () => {
    render(<IdColumnField dataset={dataset({})} />);
    expect(screen.getByRole("combobox", { name: "Identifier column" })).toHaveTextContent("name");
  });

  it("shows a viewer the value only", () => {
    render(<IdColumnField dataset={dataset({ can_edit: false })} />);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByText("name")).toBeInTheDocument();
  });

  it("says when there is none", () => {
    render(<IdColumnField dataset={dataset({ can_edit: false, id_column: null })} />);
    expect(screen.getByText("None")).toBeInTheDocument();
  });
});
