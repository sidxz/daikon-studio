import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { AttentionFigure } from "./attention";
import { BoostingFigure } from "./boosting";
import { BootstrapFigure } from "./bootstrap";
import { DomainFigure } from "./domain";
import { ForestFigure } from "./forest";
import { GaussianProcessFigure } from "./gaussian-process";
import { MessagePassingFigure } from "./message-passing";
import { SplitFigure } from "./split";

const FIGURES: [string, (t: number) => ReactElement][] = [
  ["forest", (t) => <ForestFigure t={t} />],
  ["boosting", (t) => <BoostingFigure t={t} />],
  ["gaussian process", (t) => <GaussianProcessFigure t={t} />],
  ["message passing", (t) => <MessagePassingFigure t={t} />],
  ["attention", (t) => <AttentionFigure t={t} />],
  ["split, scaffold", (t) => <SplitFigure t={t} strategy="scaffold" />],
  ["split, random", (t) => <SplitFigure t={t} strategy="random" />],
  ["bootstrap, inside", (t) => <BootstrapFigure t={t} baseline={0.7} />],
  ["bootstrap, outside", (t) => <BootstrapFigure t={t} baseline={0.55} />],
  ["domain", (t) => <DomainFigure t={t} threshold={0.3} />],
];

describe.each(FIGURES)("%s figure", (_, draw) => {
  it.each([0, 0.37, 1, 1.0001])("draws a labeled image with no NaN at t=%s", (t) => {
    const { container } = render(draw(t));
    expect(screen.getByRole("img").getAttribute("aria-label")).toBeTruthy();
    expect(container.innerHTML).not.toMatch(/NaN|Infinity|undefined/);
  });
});

describe("BootstrapFigure", () => {
  it("says within noise when the baseline is inside the interval", () => {
    render(<BootstrapFigure t={1} baseline={0.7} />);
    expect(screen.getByText("Within noise")).toBeInTheDocument();
  });
  it("says beats the baseline when it is outside", () => {
    render(<BootstrapFigure t={1} baseline={0.55} />);
    expect(screen.getByText("Beats the baseline")).toBeInTheDocument();
  });
});

describe("GaussianProcessFigure labels", () => {
  it("keeps the two annotations apart, even on a narrow engine card", () => {
    const { container } = render(<GaussianProcessFigure t={1} />);
    const texts = [...container.querySelectorAll("text")];
    const span = (el: Element) => {
      const x = Number(el.getAttribute("x"));
      const width = (el.textContent ?? "").length * 6.1;
      const anchor = el.getAttribute("text-anchor") ?? "start";
      const left = anchor === "middle" ? x - width / 2 : anchor === "end" ? x - width : x;
      return [left, left + width];
    };
    const narrow = span(texts.find((t) => t.textContent === "narrow: near data") as Element);
    const wide = span(texts.find((t) => t.textContent === "wide: no data nearby") as Element);
    expect(wide[0] - narrow[1]).toBeGreaterThanOrEqual(20);
  });
});
