import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EngineExplainer } from "./engine-explainer";

describe("EngineExplainer", () => {
  it.each([
    ["ecfp4-randomforest", /decision trees/i],
    ["ecfp4-lightgbm", /gradient boosting/i],
    ["tanimoto-gp", /gaussian process/i],
    ["chemprop-dmpnn", /seven-node graph/i],
    ["molformer-xl", /seven tokens/i],
  ])("shows the right figure for %s", (id, label) => {
    render(<EngineExplainer engineId={id} />);
    expect(screen.getByRole("img", { name: label })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /how it learns/i })).toBeInTheDocument();
  });

  it("keeps an engine figure at engine-card width, even in a wide form", () => {
    render(<EngineExplainer engineId="tanimoto-gp" />);
    const figure = screen.getByRole("img", { name: /gaussian process/i });
    expect(figure.closest(".max-w-\\[440px\\]")).not.toBeNull();
  });

  it("renders nothing for an engine with no figure", () => {
    const { container } = render(<EngineExplainer engineId="some-future-engine" />);
    expect(container).toBeEmptyDOMElement();
  });
});
