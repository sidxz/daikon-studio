/**
 * What the tool catalogue must never get wrong.
 *
 * Two things. It must not re-sort the backend's order, which is the order a
 * chemist performs these steps and is the only thing making the page read as a
 * piece of work. And it must say which tools return options rather than answers,
 * because a reader who takes a proposed core as decided has been misled by the
 * page rather than by the chemistry.
 */

import type { Tool } from "@/features/tools";
import { ToolCatalogue } from "@/features/tools";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

function tool(overrides: Partial<Tool>): Tool {
  return {
    id: "standardize",
    version: "1",
    name: "Standardize structures",
    description: "Strips salts and normalizes charges.",
    area: "data",
    inputs: [{ key: "compounds", kind: "compound_table", label: "Compounds", optional: false }],
    outputs: [{ key: "compounds", kind: "compound_table", label: "Compounds", optional: false }],
    library: "chembl_structure_pipeline",
    judgment: false,
    resolves_to: null,
    surface: "none",
    external: false,
    expands_to: [],
    ...overrides,
  };
}

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("the tool catalogue", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("keeps the backend's order, which is the order a chemist works in", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({ id: "standardize", name: "Standardize structures" }),
      tool({ id: "decompose_rgroups", name: "Decompose R groups" }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });

    const headings = await screen.findAllByRole("heading", { level: 2 });
    expect(headings.map((node) => node.textContent)).toEqual([
      "Standardize structures",
      "Decompose R groups",
    ]);
  });

  it("marks a tool that returns options rather than an answer", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({
        id: "find_cores",
        name: "Propose cores",
        judgment: true,
        surface: "editor",
        resolves_to: "core",
        outputs: [
          { key: "candidates", kind: "core_candidates", label: "Candidates", optional: false },
        ],
      }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });

    // Await the card itself. "Needs a chemist" also appears in the page's intro,
    // which renders during loading, so awaiting that text passes before the data
    // has arrived and every later assertion runs against a skeleton.
    await screen.findByRole("heading", { name: "Propose cores" });
    expect(screen.getByTitle(/You pick or draw the answer/)).toHaveTextContent("Needs a chemist");
    // The sentence wraps the resolved kind in its own element, so match on the
    // paragraph's whole text rather than a single text node.
    expect(
      screen.getByText(
        (_content, element) =>
          element?.tagName === "P" &&
          (element.textContent ?? "").includes("Nothing computes it for you"),
      ),
    ).toBeInTheDocument();
  });

  it("filters to the steps that need a person", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({ id: "standardize", name: "Standardize structures" }),
      tool({ id: "find_cores", name: "Propose cores", judgment: true, surface: "editor" }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });
    await screen.findByRole("heading", { name: "Propose cores" });

    fireEvent.click(screen.getByRole("button", { name: "Needs a chemist" }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Standardize structures" })).toBeNull();
    });
    expect(screen.getByRole("heading", { name: "Propose cores" })).toBeInTheDocument();
  });

  it("shows an optional input as optional", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({
        id: "structural_alerts",
        name: "Structural alerts",
        inputs: [
          { key: "compounds", kind: "compound_table", label: "Compounds", optional: false },
          { key: "table", kind: "rgroup_table", label: "SAR table", optional: true },
        ],
      }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });

    expect(await screen.findByTitle(/works without this/)).toHaveTextContent("SAR table");
  });
});

describe("the area tabs", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("offers a tab per area that has tools, and none for an area that does not", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({ id: "standardize", name: "Standardize structures", area: "data" }),
      tool({ id: "find_cores", name: "Propose cores", area: "sar" }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });
    await screen.findByRole("heading", { name: "Propose cores" });

    expect(screen.getByRole("tab", { name: /SAR analysis/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Structures & properties/ })).toBeInTheDocument();
    // Nothing is a protein-structure tool yet, and an always-empty tab reads as
    // something broken rather than as something coming later.
    expect(screen.queryByRole("tab", { name: /Protein structure/ })).toBeNull();
  });

  it("shows only that area's tools once a tab is chosen", async () => {
    vi.mocked(customInstance).mockResolvedValue([
      tool({ id: "standardize", name: "Standardize structures", area: "data" }),
      tool({ id: "find_cores", name: "Propose cores", area: "sar" }),
    ]);
    render(<ToolCatalogue />, { wrapper: Wrapper });
    await screen.findByRole("heading", { name: "Propose cores" });

    // Radix activates a tab on pointer-down, not on a synthetic click.
    fireEvent.mouseDown(screen.getByRole("tab", { name: /SAR analysis/ }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Standardize structures" })).toBeNull();
    });
    expect(screen.getByRole("heading", { name: "Propose cores" })).toBeInTheDocument();
  });
});
