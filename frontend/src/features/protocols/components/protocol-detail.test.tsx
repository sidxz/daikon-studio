import { ApiError } from "@/shared/lib/api/custom-instance";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProtocolDetail } from "./protocol-detail";

const hoisted = vi.hoisted(() => ({
  result: { current: {} as Record<string, unknown> },
}));

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
  useProtocol: () => hoisted.result.current,
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

beforeEach(() => {
  hoisted.result.current = { data: protocol, isLoading: false, isError: false };
});

describe("load errors", () => {
  const failed = (status: number) => {
    hoisted.result.current = {
      data: undefined,
      isLoading: false,
      isError: true,
      error: new ApiError("nope", status, undefined),
    };
    render(<ProtocolDetail protocolId="p1" />);
  };

  it("says a 404 does not exist in this workspace", () => {
    failed(404);
    expect(screen.getByText("This protocol does not exist in this workspace.")).toBeInTheDocument();
  });

  it("keeps the generic message for other errors", () => {
    failed(500);
    expect(screen.getByText("Could not load this protocol")).toBeInTheDocument();
  });
});

describe("predicted readouts", () => {
  it("states the tuned cutoff of a class readout, and only that readout's", () => {
    render(<ProtocolDetail protocolId="p1" />);
    expect(screen.getAllByText(/class at cutoff/)).toHaveLength(1);
    expect(screen.getByText("class at cutoff 0.031")).toBeInTheDocument();
  });
});
