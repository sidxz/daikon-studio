/**
 * What the steps must never get wrong: which step a press runs, and which candidate a choice
 * names. One play control per step, addressed by the step's id rather than its place.
 */

import { StudySteps } from "@/features/studies";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Wrapper, requests, route, step, study } from "./study.fixture";

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
    const served = study({
      steps: [step(), step({ id: "st2", position: 1, name: "Profile properties" })],
    });
    route({
      "GET /studies/s1": served,
      "GET /studies/s1/chat": thread,
      "POST /studies/s1/steps/st2/run": { ...served, running: true },
    });
    render(<StudySteps studyId="s1" onTalk={() => {}} />, { wrapper: Wrapper });

    expect(await screen.findAllByRole("button", { name: /^Run step \d$/ })).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Run step 2" }));

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
