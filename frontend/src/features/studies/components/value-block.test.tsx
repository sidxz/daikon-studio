import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
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
      missing_residues: ["MET A1"],
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
    // Named, not counted: a gap at the N-terminus borders nothing, one on a loop beside a
    // site makes every cavity there suspect, and a count alone cannot tell them apart.
    expect(screen.getByText("Not located in the experiment").nextSibling).toHaveTextContent(
      "MET A1",
    );
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

  it("describes a predicted result with no sites in the predictor's terms", async () => {
    const empty = {
      count: 0,
      numbering: { checked: false, reason: "no UniProt sequence to compare against" },
      structure_id: "1ubq",
      receptor_ligands: [],
      ignored_ligands: [],
      top: null,
      ligand_site_ranks: [],
    };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v5",
      key: "pocket_candidates",
      kind: "pocket_candidates",
      summary: empty,
      data: { meta: empty, pockets: [] },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v5", key: "pocket_candidates" } as never} />, {
      wrapper: Wrapper,
    });

    expect(await screen.findByText(/0 pockets\./)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/fpocket|cofactor|Monte Carlo/);
  });
});

const SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>';
const core = (rank: number, coverage: number, variable: Record<string, string>) => ({
  rank,
  smarts: "c1ccccc1",
  svg: SVG,
  coverage,
  n_atoms: 6,
  n_ring_atoms: 6,
  n_variable: 1,
  variable,
  n_rgroups: 1,
  unmatched: rank === 2 ? ["m9"] : [],
  misses: rank === 2 ? [{ id: "m9", reason: "no ring" }] : [],
  description: [`Core ${rank}`],
  preset_label: "default",
  derived: false,
  timed_out: false,
  series: null,
});
const CORES = {
  meta: { count: 2 },
  candidates: [core(1, 0.92, { X1: "O/S" }), core(2, 0.5, { X1: "N" })],
};
const FIT = {
  meta: {
    ok: true,
    reason: null,
    activity_column: "pIC50",
    n_compounds: 12,
    n_excluded_censored: 1,
    n_excluded_incomplete: 2,
    n_terms: 2,
    r2: 0.81,
    q2: 0.55,
    reference: { R1: "a", R2: "x" },
    varying_positions: ["X1", "X2"],
  },
  terms: [{ position: "X1", group: "b", n: 4, coef: 0.4, reference: false }],
  fitted: [{ id: "m1", fitted: 6.1 }],
  unmade: [{ groups: { X1: "c", X2: "z" }, prediction: 6.5 }],
};

describe("the cores block", () => {
  it("lists every candidate, draws it, and approves the selected rank", async () => {
    const onApprove = vi.fn();
    vi.mocked(customInstance).mockResolvedValue({
      id: "v1",
      key: "core_candidates",
      kind: "core_candidates",
      summary: CORES.meta,
      data: CORES,
    });
    render(
      <ValueBlock
        studyId="s1"
        value={{ id: "v1", key: "core_candidates" } as never}
        onApprove={onApprove}
      />,
      { wrapper: Wrapper },
    );
    expect(await screen.findByText(/2 candidate cores/)).toBeInTheDocument();
    expect(screen.getByText(/92%/)).toBeInTheDocument();
    expect(screen.getByText(/X1: O\/S/)).toBeInTheDocument();
    const drawings = screen.getAllByAltText("Core drawing");
    expect(drawings.length).toBeGreaterThanOrEqual(2);
    // An image, never injected markup: scripts do not run in an SVG loaded this way.
    for (const drawing of drawings) {
      expect(drawing.tagName).toBe("IMG");
      expect(drawing.getAttribute("src")).toMatch(/^data:image\/svg\+xml/);
    }
    fireEvent.click(screen.getByRole("button", { name: /rank 2/i }));
    fireEvent.click(screen.getByRole("button", { name: /use this one/i }));
    expect(onApprove).toHaveBeenCalledWith(2);
  });

  it("shows an approved core and says when it came from an earlier result", async () => {
    vi.mocked(customInstance).mockResolvedValue({
      id: "v2",
      key: "core",
      kind: "core",
      summary: { rank: 2, from: "old" },
      data: { meta: { rank: 2, from: "old" }, ...CORES.candidates[1] },
    });
    render(<ValueBlock studyId="s1" value={{ id: "v2", key: "core" } as never} earlier />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByText(/Approved on an earlier result: rank 2/)).toBeInTheDocument();
  });
});

describe("the SAR table", () => {
  it("draws a substituent with no name and writes the rest as text", async () => {
    const rows = [
      { id: "m1", R1: "Me", R1_smiles: "C", R1_svg: null, pIC50: 6.1 },
      { id: "m2", R1: "C1CC1N", R1_smiles: "C1CC1N", R1_svg: SVG, pIC50: 7.2 },
    ];
    vi.mocked(customInstance).mockResolvedValue({
      id: "v6",
      key: "sar_table",
      kind: "rgroup_table",
      summary: { columns: ["id", "R1", "R1_smiles", "R1_svg", "pIC50"], rows: 2 },
      data: rows,
    });
    render(<ValueBlock studyId="s1" value={{ id: "v6", key: "sar_table" } as never} />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByAltText("R1 drawing")).toBeInTheDocument();
    expect(screen.getAllByAltText("R1 drawing")).toHaveLength(1);
    expect(screen.getByRole("cell", { name: "Me" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "pIC50" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "R1_svg" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "R1_smiles" })).not.toBeInTheDocument();
  });
});

describe("the model fit block", () => {
  it("shows the fit, its terms and its unmade combinations", async () => {
    vi.mocked(customInstance).mockResolvedValue({
      id: "v3",
      key: "model_fit",
      kind: "model_fit",
      summary: FIT.meta,
      data: FIT,
    });
    render(<ValueBlock studyId="s1" value={{ id: "v3", key: "model_fit" } as never} />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByText(/q²/)).toBeInTheDocument();
    expect(screen.getByText("R1 a, R2 x")).toBeInTheDocument();
    expect(screen.getByText("Left out for a missing group")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "b" })).toBeInTheDocument();
    expect(screen.getByText(/c.*z/)).toBeInTheDocument();
  });

  it("leaves out an empty reference and a zero count of incomplete compounds", async () => {
    const bare = { ...FIT, meta: { ...FIT.meta, reference: {}, n_excluded_incomplete: 0 } };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v5",
      key: "model_fit",
      kind: "model_fit",
      summary: bare.meta,
      data: bare,
    });
    render(<ValueBlock studyId="s1" value={{ id: "v5", key: "model_fit" } as never} />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByText(/q²/)).toBeInTheDocument();
    expect(screen.queryByText("Reference")).not.toBeInTheDocument();
    expect(screen.queryByText("Left out for a missing group")).not.toBeInTheDocument();
  });

  it("gives only the reason when the fit was refused", async () => {
    const refused = { meta: { ok: false, reason: "Too few compounds to fit." }, terms: [] };
    vi.mocked(customInstance).mockResolvedValue({
      id: "v4",
      key: "model_fit",
      kind: "model_fit",
      summary: refused.meta,
      data: refused,
    });
    render(<ValueBlock studyId="s1" value={{ id: "v4", key: "model_fit" } as never} />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByText("Too few compounds to fit.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
