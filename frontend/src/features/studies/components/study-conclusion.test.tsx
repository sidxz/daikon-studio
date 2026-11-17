import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StudyConclusion } from "./study-conclusion";

const turns = vi.hoisted(() => ({ value: [] as unknown[] }));
vi.mock("../hooks/use-studies", () => ({ useThread: () => ({ data: { turns: turns.value } }) }));

describe("what the study found", () => {
  it("shows nothing at all until a turn carries a conclusion", () => {
    turns.value = [{ say: "Step 4 is in.", conclusion: "" }];
    const { container } = render(<StudyConclusion studyId="x" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the latest, so running on past a finished study rewrites rather than repeats", () => {
    turns.value = [
      { say: "", conclusion: "Aim at the cleft beside the nicotinamide." },
      { say: "", conclusion: "" },
      { say: "", conclusion: "Aim at Met103, Met161, Ala198 and Ile202." },
    ];
    render(<StudyConclusion studyId="x" />);
    expect(screen.getByText(/Met103, Met161/)).toBeInTheDocument();
    expect(screen.queryByText(/beside the nicotinamide/)).not.toBeInTheDocument();
  });
});
