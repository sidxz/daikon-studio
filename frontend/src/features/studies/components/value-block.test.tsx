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

    // The structure those pockets were found in was stripped of every ligand: it says so,
    // rather than listing the entry's zinc as if it stayed.
    const prepared = {
      id: "1CA2",
      ligands: [{ comp_id: "ZN", chain: "A", resseq: 262, contacts: [] }],
      kept_ligands: [],
    };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v3",
      key: "structure",
      kind: "structure",
      summary: prepared,
      data: { format: "pdb", meta: prepared },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v3", key: "structure" } as never} />, {
      wrapper: Wrapper,
    });

    expect(await screen.findByText("Kept in the receptor")).toBeInTheDocument();
    expect(screen.getByText("Ligands before preparation")).toBeInTheDocument();
    expect(screen.getByText("Kept in the receptor").nextSibling).toHaveTextContent("None");
  });

  it("names a predicted site's score for what it is and leaves its absent volume blank", async () => {
    const predicted = {
      count: 1,
      numbering: { checked: false, reason: "no UniProt sequence to compare against" },
      structure_id: "4trn",
      receptor_ligands: [],
      ignored_ligands: ["NAD"],
      top: { rank: 1, probability: 0.81 },
      ligand_site_ranks: [],
    };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v4",
      key: "pocket_candidates",
      kind: "pocket_candidates",
      summary: predicted,
      data: {
        meta: predicted,
        pockets: [
          {
            rank: 1,
            score: 17.9,
            probability: 0.81,
            residues,
            lines_ligand: [],
            ligand_overlap: [],
            annotated: [],
            interface: false,
          },
        ],
      },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v4", key: "pocket_candidates" } as never} />, {
      wrapper: Wrapper,
    });

    expect(await screen.findByText("Probability")).toBeInTheDocument();
    expect(screen.queryByText("Druggability")).toBeNull();
    expect(screen.getByText(/Ignored by the predictor: NAD/)).toBeInTheDocument();
    const detail = screen.getByText(/probability 0\.81/);
    expect(detail).toHaveTextContent("— Å³");
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
    // No other tool's name, and no cavity search's account of cofactors and volumes.
    expect(document.body.textContent).not.toMatch(/fpocket|cofactor|Monte Carlo/);
    expect(screen.getByText(/The predictor ran on the prepared structure/)).toBeInTheDocument();
  });
});
