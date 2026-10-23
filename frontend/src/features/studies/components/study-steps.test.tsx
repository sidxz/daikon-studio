/**
 * What the steps must never get wrong: which step a press runs, and which candidate a choice
 * names. One play control per step, addressed by the step's id rather than its place.
 */

import { StudySteps } from "@/features/studies";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Wrapper, requests, route, step, study, turn } from "./study.fixture";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

const thread = { summary: "", turns: [] };

const pocket = (rank: number) => ({
  rank,
  score: 12.5,
  druggability: 0.8,
  volume: 410.2,
  residues: [{ chain: "A", resname: "HIS", resseq: 93 + rank }],
  ligand_overlap: [],
  annotated: [],
  interface: false,
});

describe("the steps", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("play runs the pressed step by id", async () => {
    // Every step has run since 00:02. Only the first step's newest look is at its current
    // result and found nothing to add: the second was looked at before it ran again, and the
    // third's look was followed by a failure to look again.
    const ran = { status: "done", started_at: "2026-10-22T00:02:00Z" };
    const served = study({
      steps: [
        step(ran),
        step({ ...ran, id: "st2", position: 1, name: "Profile properties" }),
        step({ ...ran, id: "st3", position: 2, name: "Flag alerts" }),
      ],
    });
    const silent = (about: string, at: string) =>
      turn({ kind: "observe", about, at, silent: true });
    route({
      "GET /studies/s1": served,
      "GET /studies/s1/chat": {
        summary: "",
        turns: [
          silent("st1", "2026-10-22T00:03:00Z"),
          silent("st2", "2026-10-22T00:01:00Z"),
          silent("st3", "2026-10-22T00:03:00Z"),
          turn({ role: "event", kind: "observe_failed", about: "st3", at: "2026-10-22T00:04:00Z" }),
        ],
      },
      "POST /studies/s1/steps/st2/run": { ...served, running: true },
    });
    render(<StudySteps studyId="s1" onTalk={() => {}} />, { wrapper: Wrapper });

    expect(await screen.findByText("Looked, nothing to add")).toBeInTheDocument();
    const cards = screen.getAllByRole("listitem");
    expect(within(cards[0]).getByText("Looked, nothing to add")).toBeInTheDocument();
    expect(screen.getAllByText("Looked, nothing to add")).toHaveLength(1);

    expect(screen.getAllByRole("button", { name: /^Run step \d again$/ })).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "Run step 2 again" }));

    await waitFor(() => expect(requests("POST", "/studies/s1/steps/st2/run")).toHaveLength(1));
    expect(requests("POST", "/studies/s1/steps/st1/run")).toHaveLength(0);
  });

  it("Use this one posts approve with the rank", async () => {
    const meta = { count: 2 };
    const served = study({
      steps: [
        step({
          tool_id: "find_pockets",
          name: "Find binding pockets",
          status: "done",
          judgment: true,
          outputs: [
            {
              id: "v1",
              key: "pocket_candidates",
              kind: "pocket_candidates",
              summary: meta,
              produced_by: "st1",
              asserted_by: null,
            },
          ],
        }),
      ],
    });
    route({
      "GET /studies/s1": served,
      "GET /studies/s1/chat": thread,
      "GET /studies/s1/values/v1": {
        id: "v1",
        key: "pocket_candidates",
        kind: "pocket_candidates",
        summary: meta,
        data: { meta, pockets: [pocket(1), pocket(2)] },
      },
      "POST /studies/s1/steps/st1/approve": served,
    });
    render(<StudySteps studyId="s1" onTalk={() => {}} />, { wrapper: Wrapper });

    fireEvent.click(
      await screen.findByRole("button", { name: "Show pocket rank 2 in the viewer" }),
    );
    expect(screen.getByText("A:HIS95")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use this one" }));

    await waitFor(() => expect(requests("POST", "/studies/s1/steps/st1/approve")).toHaveLength(1));
    expect(requests("POST", "/studies/s1/steps/st1/approve")[0].data).toEqual({ candidate: 2 });
  });
});
