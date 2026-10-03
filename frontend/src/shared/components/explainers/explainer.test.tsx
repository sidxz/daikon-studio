import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Explainer } from "./explainer";
import { useExplainerStore } from "./explainer-store";
import { useTimeline } from "./use-timeline";

let frames: FrameRequestCallback[] = [];
let observe: IntersectionObserverCallback | null = null;
const disconnect = vi.fn();
const flush = (now: number) =>
  act(() => {
    const run = frames;
    frames = [];
    for (const cb of run) cb(now);
  });
const enterView = () =>
  act(() =>
    observe?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver),
  );
const reduceMotion = (on: boolean) =>
  vi.spyOn(window, "matchMedia").mockImplementation((q: string) => ({
    matches: on && q.includes("reduce"),
    media: q,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }));

beforeEach(() => {
  frames = [];
  observe = null;
  disconnect.mockClear();
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(cb: IntersectionObserverCallback) {
        observe = cb;
      }
      observe() {}
      unobserve() {}
      disconnect() {
        disconnect();
      }
    },
  );
  reduceMotion(false);
  useExplainerStore.setState({ closed: {} });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function Probe() {
  const { ref, t } = useTimeline<HTMLDivElement>(1000);
  return <div ref={ref} data-testid="probe" data-t={t} />;
}
const tOf = () => Number(screen.getByTestId("probe").dataset.t);

describe("useTimeline", () => {
  it("rests on the final frame until the figure is in view", () => {
    render(<Probe />);
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
  });

  it("plays once from 0 to 1 when in view, then stops observing", () => {
    render(<Probe />);
    enterView();
    expect(tOf()).toBe(0);
    flush(100);
    flush(600);
    expect(tOf()).toBeCloseTo(0.5);
    flush(1200);
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
    expect(disconnect).toHaveBeenCalled();
  });

  it("never animates under reduced motion", () => {
    reduceMotion(true);
    render(<Probe />);
    enterView();
    expect(tOf()).toBe(1);
    expect(frames).toHaveLength(0);
  });

  it("replays on pointer enter when idle", () => {
    render(<Probe />);
    fireEvent.pointerEnter(screen.getByTestId("probe"));
    expect(tOf()).toBe(0);
  });
});

describe("Explainer", () => {
  const renderIt = () =>
    render(
      <Explainer id="forest" caption="Trees vote." durationMs={1000}>
        {(t) => <svg role="img" aria-label="figure" data-t={t} />}
      </Explainer>,
    );

  it("is open on first sight", () => {
    renderIt();
    expect(screen.getByRole("img", { name: "figure" })).toBeInTheDocument();
    expect(screen.getByText("Trees vote.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /replay/i })).toBeInTheDocument();
  });

  it("remembers being closed, per figure id", () => {
    const { unmount } = renderIt();
    fireEvent.click(screen.getByRole("button", { name: /how this works/i }));
    expect(useExplainerStore.getState().closed.forest).toBe(true);
    unmount();
    renderIt();
    expect(screen.queryByRole("img", { name: "figure" })).not.toBeInTheDocument();
  });

  it("plays when reopened after being closed", () => {
    useExplainerStore.setState({ closed: { forest: true } });
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /how this works/i }));
    enterView();
    expect(Number(screen.getByRole("img", { name: "figure" }).dataset.t)).toBe(0);
  });
});
