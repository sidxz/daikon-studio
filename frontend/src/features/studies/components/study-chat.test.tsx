/**
 * The one thread, as the rail shows it: what the agent did and on whose words, a proposal that
 * waits for a press, and a message sent about one step.
 */

import { StudyChat } from "@/features/studies";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Wrapper, requests, route, step, study, turn } from "./study.fixture";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

const planned = study({
  steps: [
    step({ tool_id: "fetch_pdb", name: "Fetch from PDB", status: "done" }),
    step({ id: "st2", position: 1, tool_id: "prepare_structure", name: "Prepare structure" }),
  ],
});

function chat(about: string | null = null) {
  return <StudyChat studyId="s1" provider="" about={about} onClearAbout={() => {}} />;
}

describe("the study's thread", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("renders a reply turn's actions with their quotes and the dropped count", async () => {
    route({
      "GET /studies/s1": planned,
      "GET /studies/s1/chat": {
        summary: "",
        turns: [
          turn({ role: "scientist", say: "Rerun without the zinc, please." }),
          turn({
            say: "The zinc is out; step 2 is waiting.",
            actions: [
              {
                does: "settings",
                step: "st2",
                detail: "Changed step 2 (Prepare structure): keep_ligands=false.",
                quoting: "Rerun without the zinc",
              },
            ],
            // About a step a later edit took out: named as gone, never a crash.
            points: [
              {
                step: "gone",
                at: "rank 2",
                label: "zinc site",
                evidence: "ligand_overlap",
                cites: [],
              },
            ],
            dropped: 2,
            written_by: "anthropic/claude-opus-5-5",
          }),
        ],
      },
    });
    render(chat(), { wrapper: Wrapper });

    expect(
      await screen.findByText(
        "Did: Changed step 2 (Prepare structure): keep_ligands=false, on your words: 'Rerun without the zinc'",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/2 claims dropped, unsupported by this result/)).toBeInTheDocument();
    expect(
      screen.getByText(/a removed step · rank 2 · zinc site · ligand overlap/),
    ).toBeInTheDocument();
  });

  it("a proposal's button posts apply and then reads as done", async () => {
    const proposal = {
      id: "p1",
      action: {
        does: "settings",
        step_id: "st2",
        tool_id: null,
        before_id: null,
        purpose: null,
        settings: { keep_ligands: false },
        gated: null,
        candidate: null,
        until_id: null,
      },
      why: "The zinc fills the bottom of the cone.",
      applied_at: null,
    };
    let applied: string | null = null;
    route({
      "GET /studies/s1": planned,
      "GET /studies/s1/chat": () => ({
        summary: "",
        turns: [
          turn({
            kind: "observe",
            about: "st2",
            say: "Rank 1 sits against the zinc.",
            proposals: [{ ...proposal, applied_at: applied }],
          }),
        ],
      }),
      "POST /studies/s1/proposals/p1/apply": () => {
        applied = "2026-10-22T00:02:00Z";
        return planned;
      },
    });
    render(chat(), { wrapper: Wrapper });

    expect(await screen.findByText("Change step 2: keep_ligands=false")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Do it" }));

    expect(
      await screen.findByText("Done. The step is waiting for you to run it."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Do it" })).toBeNull();
    expect(requests("POST", "/studies/s1/proposals/p1/apply")).toHaveLength(1);
  });

  it("an about chip is sent with the message and can be cleared", async () => {
    const thread = { summary: "", turns: [turn({ say: "Step 1 is done." })] };
    route({
      "GET /studies/s1": planned,
      "GET /studies/s1/chat": thread,
      "POST /studies/s1/chat": thread,
    });
    function Rail() {
      const [about, setAbout] = useState<string | null>("st2");
      return (
        <StudyChat studyId="s1" provider="" about={about} onClearAbout={() => setAbout(null)} />
      );
    }
    render(<Rail />, { wrapper: Wrapper });

    expect(await screen.findByText("About step 2 (Prepare structure)")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Message to Daikon Studio"), {
      target: { value: "Why keep the zinc?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(requests("POST", "/studies/s1/chat")).toHaveLength(1));
    expect(requests("POST", "/studies/s1/chat")[0].data).toEqual({
      message: "Why keep the zinc?",
      about: "st2",
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear the step" }));
    expect(screen.queryByText("About step 2 (Prepare structure)")).toBeNull();
  });
});
