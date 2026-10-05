import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProtocolDetail } from "./protocol-detail";

// Only the readouts card is under test.
const protocol = {
  id: "p1",
  name: "Reactivity",
  status: "draft",
  engine_id: "ecfp4-randomforest",
  dataset_id: "d1",
  can_delete: false,
  readouts: [
    {
      name: "reactive",
      type: "class",
      unit: null,
      direction: null,
      description: "",
      threshold: 0.031,
    },
    { name: "toxic", type: "class", unit: null, direction: null, description: "", threshold: null },
    {
      name: "logS",
      type: "numeric",
      unit: "log mol/L",
      direction: null,
      description: "",
      threshold: null,
    },
  ],
};

vi.mock("../hooks/use-protocols", () => ({
  useProtocol: () => ({ data: protocol, isLoading: false, isError: false }),
  useScorecard: () => ({ isLoading: false, isError: false, data: undefined }),
  usePublishProtocol: () => ({ mutate: vi.fn(), isPending: false }),
}));
vi.mock("@/features/datasets", () => ({ useDataset: () => ({ data: undefined }) }));
vi.mock("@/shared/lib/auth/use-workspace-members", () => ({
  useMemberName: () => () => undefined,
}));
vi.mock("@/shared/lib/stores/breadcrumb-store", () => ({ useBreadcrumbTrail: () => undefined }));
vi.mock("./delete-protocol-button", () => ({ DeleteProtocolButton: () => null }));
vi.mock("./protocol-chemical-space", () => ({ ProtocolChemicalSpace: () => null }));
vi.mock("./protocol-runs", () => ({ ProtocolRuns: () => null }));

describe("predicted readouts", () => {
  it("states the tuned cutoff of a class readout, and only that readout's", () => {
    render(<ProtocolDetail protocolId="p1" />);
    expect(screen.getAllByText(/class at cutoff/)).toHaveLength(1);
    expect(screen.getByText("class at cutoff 0.031")).toBeInTheDocument();
  });
});
