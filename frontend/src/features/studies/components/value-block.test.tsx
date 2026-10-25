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
    expect(
      screen.getByText("Residues unresolved in the experimental structure").nextSibling,
    ).toHaveTextContent("MET A1");
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
    expect(
      screen.getByText(/The predictor used only protein atoms from the prepared structure/),
    ).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: /select this core/i }));
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
    expect(screen.getByText("Excluded: missing substituent assignments")).toBeInTheDocument();
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
    expect(screen.queryByText("Excluded: missing substituent assignments")).not.toBeInTheDocument();
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

describe("a cell that is a payload rather than a value", () => {
  it("reports a docked pose's size instead of printing its coordinates", async () => {
    // A pose is kilobytes of coordinates per row. Printed, it buries every column beside
    // it and tells the reader nothing; the geometry stays in the value, not in the cell.
    const pose = "ATOM      1  C   UNL     1      17.229  19.516  10.262\n".repeat(60);
    vi.mocked(customInstance).mockResolvedValue({
      id: "v9",
      key: "poses",
      kind: "poses",
      summary: { columns: ["compound", "affinity_kcal_mol", "pose_pdbqt"], rows: 1 },
      data: [{ compound: "indinavir", affinity_kcal_mol: -11.5, pose_pdbqt: pose }],
    });
    render(<ValueBlock studyId="s1" value={{ id: "v9", key: "poses" } as never} />, {
      wrapper: Wrapper,
    });
    expect(await screen.findByRole("cell", { name: "indinavir" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "-11.5" })).toBeInTheDocument();
    expect(screen.queryByText(/ATOM/)).not.toBeInTheDocument();
    expect(screen.getByText(/kB, not shown/)).toBeInTheDocument();
  });
});

describe("the contacts a pose makes", () => {
  it("groups a pose's contacts by residue and puts the hydrogen bonds first", async () => {
    // One row per contact is the shape the tool returns and the shape nobody can hold
    // against the literature. The question is which residues the pose reaches, and the
    // directional contacts are the ones a reader weighs first.
    vi.mocked(customInstance).mockResolvedValue({
      id: "v10",
      key: "interactions",
      kind: "interactions",
      summary: { columns: ["compound", "pose_rank", "residue", "chain", "interaction"], rows: 4 },
      data: [
        {
          compound: "indinavir",
          pose_rank: 1,
          residue: "ILE50",
          chain: "B",
          interaction: "VdWContact",
        },
        {
          compound: "indinavir",
          pose_rank: 1,
          residue: "ASP25",
          chain: "A",
          interaction: "HBDonor",
        },
        {
          compound: "indinavir",
          pose_rank: 1,
          residue: "ASP25",
          chain: "A",
          interaction: "Hydrophobic",
        },
        {
          compound: "indinavir",
          pose_rank: 1,
          residue: "GLY48",
          chain: "A",
          interaction: "Hydrophobic",
        },
      ],
    });
    render(<ValueBlock studyId="s1" value={{ id: "v10", key: "interactions" } as never} />, {
      wrapper: Wrapper,
    });
    // Three residues from four contacts: ASP25 is one residue making two kinds.
    expect(await screen.findByText(/3 residues/)).toBeInTheDocument();
    // A chip per residue, hydrogen bond first, then the two that only pack or touch.
    const chips = [...document.querySelectorAll('[data-slot="badge"]')].map((c) => c.textContent);
    expect(chips).toEqual([
      "ASP25 AHBDonor / Hydrophobic",
      "GLY48 AHydrophobic",
      "ILE50 BVdWContact",
    ]);
  });
});

describe("reading a prepared structure before docking into it", () => {
  it("names the residues whose protonation came out against expectation", async () => {
    // The card is the gate. Behind this summary sit fifty-odd pKa records, which nobody
    // reads; the aspartate left neutral is the one thing that changes what the scores mean.
    vi.mocked(customInstance).mockResolvedValue({
      id: "v11",
      key: "receptor",
      kind: "structure",
      summary: {
        id: "1HSG",
        chains: ["A", "B"],
        kept_ligands: [],
        prepared: true,
        ph: 7.0,
        protonation_shifted: ["ASP25 A pKa 7.9, neutral (usually charged)"],
        protonation:
          "Assigned from estimated pKa at pH 7.0. Titratable groups not named above took the state their kind usually has at this pH.",
      },
      data: {},
    });
    render(<ValueBlock studyId="s1" value={{ id: "v11", key: "receptor" } as never} />, {
      wrapper: Wrapper,
    });
    expect(
      await screen.findByText("ASP25 A pKa 7.9, neutral (usually charged)"),
    ).toBeInTheDocument();
    expect(screen.getByText(/took the state their kind usually has/)).toBeInTheDocument();
  });
});

describe("judging a grouping", () => {
  it("draws clustered compounds grouped, with the centroid marked", async () => {
    vi.mocked(customInstance).mockResolvedValue({
      id: "v10",
      key: "compounds",
      kind: "compound_table",
      summary: { columns: ["compound_id", "smiles", "cluster", "is_centroid"], rows: 3 },
      data: [
        { compound_id: "C1", smiles: "CCO", cluster: 1, is_centroid: true },
        { compound_id: "C2", smiles: "CCC", cluster: 1, is_centroid: false },
        { compound_id: "C3", smiles: "c1ccccc1", cluster: 2, is_centroid: true },
      ],
    });
    render(<ValueBlock studyId="s1" value={{ id: "v10", key: "compounds" } as never} />, {
      wrapper: Wrapper,
    });
    // Two clusters, the larger one first; a column of integers is not a reviewable answer.
    expect(await screen.findByText(/Cluster 1 · 2 compounds/)).toBeInTheDocument();
    expect(screen.getByText(/Cluster 2 · 1 compound$/)).toBeInTheDocument();
  });
});
