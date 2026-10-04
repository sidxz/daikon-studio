import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { AttentionFigure } from "./attention";
import { BoostingFigure } from "./boosting";
import { type BootstrapData, BootstrapFigure } from "./bootstrap";
import { DomainFigure } from "./domain";
import { ForestFigure } from "./forest";
import { GaussianProcessFigure } from "./gaussian-process";
import { MessagePassingFigure } from "./message-passing";
import { SplitFigure } from "./split";

/** RMSE redraws around 0.72, the shape of a real scaffold-split card. */
const RMSE: BootstrapData = {
  metric: "RMSE",
  higherIsBetter: false,
  interval: [0.586, 0.864],
  baseline: 1.248,
  redraws: {
    edges: [0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.9],
    counts: [20, 130, 330, 340, 160, 20],
  },
  compounds: Array.from({ length: 197 }, (_, i) => ({
    actual: i % 5,
    predicted: (i % 5) + (i % 3) * 0.4,
  })),
  testSize: 197,
  cutoff: null,
};
const MCC: BootstrapData = {
  ...RMSE,
  metric: "MCC",
  higherIsBetter: true,
  baseline: 0.7,
  interval: [0.62, 0.86],
  compounds: Array.from({ length: 60 }, (_, i) => ({ actual: i % 2, predicted: (i % 7) / 6 })),
  testSize: 60,
  cutoff: 0.5,
};

const FIGURES: [string, (t: number) => ReactElement][] = [
  ["forest", (t) => <ForestFigure t={t} />],
  ["boosting", (t) => <BoostingFigure t={t} />],
  ["gaussian process", (t) => <GaussianProcessFigure t={t} />],
  ["message passing", (t) => <MessagePassingFigure t={t} />],
  ["attention", (t) => <AttentionFigure t={t} />],
  ["split, scaffold", (t) => <SplitFigure t={t} strategy="scaffold" />],
  ["split, random", (t) => <SplitFigure t={t} strategy="random" />],
  ["bootstrap, regression", (t) => <BootstrapFigure t={t} data={RMSE} />],
  ["bootstrap, classification", (t) => <BootstrapFigure t={t} data={MCC} />],
  ["bootstrap, off scale", (t) => <BootstrapFigure t={t} data={{ ...RMSE, baseline: 9 }} />],
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
  it("reads the real interval and says a baseline above an RMSE interval is beaten", () => {
    render(<BootstrapFigure t={1} data={RMSE} />);
    expect(screen.getByText("0.586 to 0.864 (95% interval)")).toBeInTheDocument();
    expect(
      screen.getByText(
        /\(1\.248\) is outside the model's likely range, so the model is genuinely better/,
      ),
    ).toBeInTheDocument();
  });
  it("says inside when the baseline is", () => {
    render(<BootstrapFigure t={1} data={MCC} />);
    expect(screen.getByText(/\(0\.700\) is inside the model's likely range/)).toBeInTheDocument();
  });
  it("says the baseline leads when it is outside on the better side", () => {
    render(<BootstrapFigure t={1} data={{ ...MCC, baseline: 0.9 }} />);
    expect(screen.getByText(/so the baseline is genuinely better/)).toBeInTheDocument();
  });
  it("names a baseline too far for the axis at its edge instead of crushing the dots", () => {
    render(<BootstrapFigure t={1} data={{ ...RMSE, baseline: 9 }} />);
    expect(screen.getByText("baseline model: 9.000 →")).toBeInTheDocument();
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
