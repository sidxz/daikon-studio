// @vitest-environment jsdom
// Renders into real jsdom — the rest of the suite runs under vitest's default
// "node" environment (see use-page-save.test.ts), so this is scoped here only.
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SequenceTrack } from "./sequence-track";

describe("SequenceTrack", () => {
  it("renders one cell per residue", () => {
    const sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL";
    const { container } = render(<SequenceTrack seqType="protein" sequence={sequence} />);
    expect(container.querySelectorAll("[data-residue]")).toHaveLength(sequence.length);
  });

  it("chunks into rows of 60 residues", () => {
    const sequence = "A".repeat(125); // 60 + 60 + 5
    const { container } = render(<SequenceTrack seqType="dna" sequence={sequence} />);
    expect(container.querySelectorAll("[data-sequence-row]")).toHaveLength(3);
  });

  it("marks exactly the residues covered by a feature span, with a hover label", () => {
    const sequence = "ACDEFGHIKLMNPQRSTVWY"; // 20 residues
    const feature = { start: 3, end: 6, type: "domain", label: "Test domain" };
    const { container } = render(
      <SequenceTrack seqType="protein" sequence={sequence} features={[feature]} />,
    );

    const marked = container.querySelectorAll('[data-residue][data-in-feature="true"]');
    expect(marked).toHaveLength(feature.end - feature.start + 1);

    // 1-based inclusive range: residues 3..6 are sequence[2..5].
    const markedText = Array.from(marked)
      .map((el) => el.textContent)
      .join("");
    expect(markedText).toBe(sequence.slice(feature.start - 1, feature.end));

    expect(marked[0]?.getAttribute("title")).toContain("Test domain");
    expect(marked[0]?.getAttribute("aria-label")).toContain("Test domain");

    // Residues outside the span carry no feature marker.
    const unmarked = container.querySelectorAll("[data-residue]:not([data-in-feature])");
    expect(unmarked).toHaveLength(sequence.length - marked.length);
  });
});
