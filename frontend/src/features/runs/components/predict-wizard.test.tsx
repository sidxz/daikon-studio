import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PredictWizard } from "./predict-wizard";

const hoisted = vi.hoisted(() => ({
  upload: vi.fn(),
  create: vi.fn(),
  push: vi.fn(),
}));

const IMPORTED = {
  upload_ref: "up-1",
  compound_count: 3,
  without_structure: 0,
  sample: ["CCO"],
  source: { protocol_name: "NadD", run_date: "2026-06-05" },
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: hoisted.push }),
  useSearchParams: () => new URLSearchParams("protocol=p1"),
}));
vi.mock("@/shared/lib/app-config", () => ({
  useAppConfig: () => ({ chemcellarUrl: "http://cellar" }),
}));
vi.mock("@/features/datasets", () => ({
  PREDICTION_TEMPLATE_CSV: "",
  useDataset: () => ({ data: undefined }),
}));
vi.mock("@/features/engines", () => ({ useEngines: () => ({ data: [] }) }));
vi.mock("@/features/protocols", () => ({
  useProtocols: () => ({
    data: { items: [{ id: "p1", name: "Protocol one", status: "published", readouts: [] }] },
  }),
}));
vi.mock("../hooks/use-runs", () => ({
  useUploadPredictionFile: () => ({ mutateAsync: hoisted.upload, isPending: false }),
  useCreateRun: () => ({ mutateAsync: hoisted.create, isPending: false }),
}));
vi.mock("./prediction-preview", () => ({ PredictionPreview: () => null }));
// The picker's own behavior is not under test: it reports an import, or null.
vi.mock("./chemcellar-picker", () => ({
  ChemCellarPicker: ({ onImported }: { onImported: (value: unknown) => void }) => (
    <>
      <button type="button" onClick={() => onImported(IMPORTED)}>
        import
      </button>
      <button type="button" onClick={() => onImported(null)}>
        clear
      </button>
    </>
  ),
}));

describe("PredictWizard, ChemCellar tab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hoisted.create.mockResolvedValue({ id: "run-1", status: "pending" });
  });

  async function openChemCellar() {
    render(<PredictWizard />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "From ChemCellar" }), { button: 0 });
    fireEvent.click(screen.getByRole("tab", { name: "From ChemCellar" }));
    return screen.findByRole("button", { name: "import" });
  }

  it("submits the import's upload without uploading a file", async () => {
    fireEvent.click(await openChemCellar());
    fireEvent.click(await screen.findByRole("button", { name: "Predict 3 compounds" }));

    await waitFor(() => expect(hoisted.create).toHaveBeenCalledTimes(1));
    expect(hoisted.create).toHaveBeenCalledWith({
      protocol_id: "p1",
      upload_ref: "up-1",
      structure_column: "smiles",
      id_column: "compound_id",
      name: "NadD · 2026-06-05",
    });
    expect(hoisted.upload).not.toHaveBeenCalled();
  });

  it("disables Predict once the picker reports nothing selected", async () => {
    fireEvent.click(await openChemCellar());
    expect(await screen.findByRole("button", { name: "Predict 3 compounds" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "clear" }));

    expect(await screen.findByRole("button", { name: "Predict 0 compounds" })).toBeDisabled();
  });
});

describe("PredictWizard, run name", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hoisted.create.mockResolvedValue({ id: "run-1", status: "pending" });
    hoisted.upload.mockResolvedValue("up-2");
  });

  it("prefills the name from the file and submits it", async () => {
    const { container } = render(<PredictWizard />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["smiles\nCCO\nCCN\n"], "batch-7.csv", { type: "text/csv" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(screen.getByLabelText(/Run name/)).toHaveValue("batch-7"));
    fireEvent.click(await screen.findByRole("button", { name: /^Predict/ }));

    await waitFor(() => expect(hoisted.create).toHaveBeenCalledTimes(1));
    expect(hoisted.create).toHaveBeenCalledWith(expect.objectContaining({ name: "batch-7" }));
  });
});
