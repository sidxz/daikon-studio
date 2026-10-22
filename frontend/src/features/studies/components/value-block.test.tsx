import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { ValueBlock } from "./value-block";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const residues = Array.from({ length: 8 }, (_, i) => ({
  chain: "A",
  resname: "LEU",
  resseq: 23 + i,
}));
const meta = {
  count: 2,
  numbering: { checked: true, matches: true, agree: 76, of: 76, offset: 0 },
  receptor_ligands: ["NAD"],
  top: { rank: 1, druggability: 0.8 },
  ligand_site_ranks: [2],
};
const pocket = (rank: number, overlap: unknown[]) => ({
  rank,
  score: 12.5,
  druggability: 0.8,
  volume: 410.2,
  alpha_spheres: 30,
  residues,
  ligand_overlap: overlap,
  annotated: rank === 2 ? [{ type: "Binding site", start: 1, end: 2, description: "" }] : [],
  interface: rank === 2,
});

describe("the pockets block", () => {
  it("states the count and the ligand ranks, and marks overlapping rows", async () => {
    vi.mocked(customInstance).mockResolvedValue({
      id: "v1",
      key: "pocket_candidates",
      kind: "pocket_candidates",
      summary: meta,
      data: {
        meta,
        pockets: [
          pocket(1, []),
          pocket(2, [
            { comp_id: "ATP", chain: "A", resseq: 400, shared: 5, of: 9, additive: false },
          ]),
        ],
      },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v1", key: "pocket_candidates" } as never} />, {
      wrapper: Wrapper,
    });

    expect(await screen.findByText(/2 pockets\. Rank 2 overlaps/)).toBeInTheDocument();
    expect(screen.getByText(/numbering agrees with UniProt at 76 of 76/)).toBeInTheDocument();
    expect(screen.getByText("ATP 5/9")).toBeInTheDocument();
    expect(screen.getByText("Binding site")).toBeInTheDocument();
    expect(screen.getAllByText("8")[0]).toHaveAttribute(
      "title",
      "A:LEU 23, A:LEU 24, A:LEU 25, A:LEU 26, A:LEU 27, A:LEU 28",
    );
    const rows = screen.getAllByRole("row");
    expect(rows[1]).not.toHaveClass("font-bold");
    expect(rows[2]).toHaveClass("font-bold");
  });

  it("marks a drop component and says why no site is labelled", async () => {
    const unchecked = {
      count: 1,
      numbering: { checked: false, reason: "no UniProt sequence to compare against" },
      receptor_ligands: [],
      top: { rank: 1, druggability: 0.8 },
      ligand_site_ranks: [],
    };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v2",
      key: "pocket_candidates",
      kind: "pocket_candidates",
      summary: unchecked,
      data: {
        meta: unchecked,
        pockets: [
          pocket(1, [
            { comp_id: "GOL", chain: "A", resseq: 300, shared: 3, of: 6, additive: true },
          ]),
        ],
      },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v2", key: "pocket_candidates" } as never} />, {
      wrapper: Wrapper,
    });

    expect(await screen.findByText("GOL (additive) 3/6")).toBeInTheDocument();
    expect(screen.getByText(/no annotated sites are labelled/)).toBeInTheDocument();
  });
});
