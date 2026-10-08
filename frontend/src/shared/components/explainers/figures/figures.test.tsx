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
import { SplitFigure, splitCaption } from "./split";

/** RMSE redraws around 0.72, the shape of a real scaffold-split card. */
const RMSE: BootstrapData = {
  mode: "score",
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

/** Model minus baseline RMSE on each redraw: a consistent lead, all below zero. */
const DIFF: BootstrapData = {
  ...RMSE,
  mode: "difference",
  interval: [-0.176, -0.139],
  baseline: 0,
  redraws: { edges: [-0.19, -0.17, -0.15, -0.13], counts: [200, 600, 200] },
};

const FIGURES: [string, (t: number) => ReactElement][] = [
  ["forest", (t) => <ForestFigure t={t} />],
  ["boosting", (t) => <BoostingFigure t={t} />],
  ["gaussian process", (t) => <GaussianProcessFigure t={t} />],
  ["message passing", (t) => <MessagePassingFigure t={t} />],
  ["attention", (t) => <AttentionFigure t={t} />],
  ["split, scaffold", (t) => <SplitFigure t={t} strategy="scaffold" />],
  ["split, random", (t) => <SplitFigure t={t} strategy="random" />],
  ["split, identity", (t) => <SplitFigure t={t} strategy="identity" />],
  ["split, position", (t) => <SplitFigure t={t} strategy="position" />],
  ["bootstrap, regression", (t) => <BootstrapFigure t={t} data={RMSE} />],
  ["bootstrap, classification", (t) => <BootstrapFigure t={t} data={MCC} />],
  ["bootstrap, off scale", (t) => <BootstrapFigure t={t} data={{ ...RMSE, baseline: 9 }} />],
  ["bootstrap, difference", (t) => <BootstrapFigure t={t} data={DIFF} />],
  ["domain", (t) => <DomainFigure t={t} threshold={0.3} />],
];

describe.each(FIGURES)("%s figure", (_, draw) => {
  it.each([0, 0.37, 1, 1.0001])("draws a labeled image with no NaN at t=%s", (t) => {
    const { container } = render(draw(t));
    expect(screen.getByRole("img").getAttribute("aria-label")).toBeTruthy();
    expect(container.innerHTML).not.toMatch(/NaN|Infinity|undefined/);
  });
});

describe("SplitFigure", () => {
  it("labels the grouped split actually in play, not scaffold by default", () => {
    render(<SplitFigure t={1} strategy="identity" />);
    expect(screen.getByText("Identity split")).toBeInTheDocument();
    expect(screen.getByText("Random split")).toBeInTheDocument();
    expect(screen.queryByText("Scaffold split")).not.toBeInTheDocument();
  });

  it("tells each split what its clusters are, and never calls a sequence a scaffold", () => {
    expect(splitCaption("scaffold")).toContain("Bemis–Murcko scaffold");
    expect(splitCaption("identity")).toContain("one protein family");
    expect(splitCaption("position")).toContain("one mutated residue position");
    expect(splitCaption("identity")).not.toContain("Bemis–Murcko");
    expect(splitCaption("position")).not.toContain("Bemis–Murcko");
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
  it("marks zero as no difference and says a lead below it holds for RMSE", () => {
    render(<BootstrapFigure t={1} data={DIFF} />);
    expect(screen.getByText("no difference")).toBeInTheDocument();
    expect(screen.getByText("likely range of the difference")).toBeInTheDocument();
    expect(
      screen.getByText(/RMSE, model minus baseline \(lower favors the model\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Every likely difference favors the model, so its lead holds up/),
    ).toBeInTheDocument();
  });
  it("says this test cannot tell when zero is inside the difference's range", () => {
    render(<BootstrapFigure t={1} data={{ ...DIFF, interval: [-0.1, 0.05] }} />);
    expect(
      screen.getByText(/Zero is inside the likely range of the difference/),
    ).toBeInTheDocument();
  });
  it("says the baseline leads when the whole range is on its side", () => {
    render(<BootstrapFigure t={1} data={{ ...DIFF, interval: [0.05, 0.1] }} />);
    expect(screen.getByText(/Every likely difference favors the baseline/)).toBeInTheDocument();
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
