/**
 * The study page from its first message: the agent's questions answered in the rail, a plan that
 * says why it cannot start, and a step whose tool can be swapped before anything runs.
 */

import { StudyPage } from "@/features/studies";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RUN_UNDER_WAY } from "../types";
import { Wrapper, requests, route, step, study, turn } from "./study.fixture";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const { customInstance } = await import("@/shared/lib/api/custom-instance");

const one = [{ provider: "anthropic", model: "claude-opus-5-5" }];
const draft = (overrides = {}) =>
  study({ status: "draft", started_at: null, steps: [], restated_goal: "", ...overrides });

describe("the study page", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("renders the question cards from the asking turn and sends the answers", async () => {
    const replaceState = vi.spyOn(window.history, "replaceState");
    const asking = turn({
      kind: "asking",
      say: "Locate the cavity where ligands bind in carbonic anhydrase II.",
      questions: [
        {
          ask: "Should the zinc stay in the structure?",
          why: "It sits at the bottom of the active-site cone.",
          options: ["Keep it", "Remove it"],
          default: null,
        },
      ],
    });
    const opened = draft({ restated_goal: asking.say });
    route({
      "GET /studies/providers": [...one, { provider: "openai", model: "gpt-5" }],
      "POST /studies": opened,
      "GET /studies/s1": opened,
      "GET /studies/s1/chat": { summary: "", turns: [asking] },
      "POST /studies/s1/chat": {
        summary: "",
        turns: [asking, turn({ role: "scientist", say: "Should the zinc stay? → Keep it" })],
      },
    });
    render(<StudyPage />, { wrapper: Wrapper });

    fireEvent.change(await screen.findByLabelText("What do you want to find out?"), {
      target: { value: "Find the binding pocket of carbonic anhydrase II, PDB 1CA2" },
    });
    fireEvent.change(await screen.findByRole("combobox", { name: "Model" }), {
      target: { value: "openai" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask Daikon Studio" }));

    // The draft is open and the address is the study's, without leaving the page.
    expect(await screen.findByText("Should the zinc stay in the structure?")).toBeInTheDocument();
    expect(replaceState.mock.calls.at(-1)?.[2]).toBe("/studies/s1");
    const form = requests("POST", "/studies")[0].data as FormData;
    expect(form.get("message")).toBe("Find the binding pocket of carbonic anhydrase II, PDB 1CA2");
    expect(form.get("provider")).toBe("openai");

    fireEvent.click(screen.getByRole("button", { name: "Keep it" }));
    expect(screen.getByRole("button", { name: "Send answers" })).toBeEnabled();
    // A note sent from the composer instead still carries the choice made on the card.
    fireEvent.change(screen.getByLabelText("Message to Daikon Studio"), {
      target: { value: "It is catalytic." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(requests("POST", "/studies/s1/chat")).toHaveLength(1));
    // The answers travel as a field beside the message, never in its prose, and the model
    // chosen on the form answers this turn too.
    expect(requests("POST", "/studies/s1/chat")[0].data).toEqual({
      message: "It is catalytic.",
      answers: [{ ask: "Should the zinc stay in the structure?", answer: "Keep it" }],
      provider: "openai",
    });
    replaceState.mockRestore();
  });

  it("disables Start while a step has a problem and says why", async () => {
    const problem = "Fetch from AlphaFold needs accession, which is not set.";
    const steps = [step({ tool_id: "fetch_alphafold", name: "Fetch from AlphaFold", problem })];
    route({
      "GET /studies/providers": one,
      "GET /studies/s1": draft({ steps }),
      "GET /studies/s1/chat": { summary: "", turns: [] },
    });
    const { unmount } = render(<StudyPage studyId="s1" />, { wrapper: Wrapper });

    expect(await screen.findByRole("button", { name: "Start this study" })).toBeDisabled();
    expect(screen.getByText(/Step 1 has a problem, shown on its card/)).toBeInTheDocument();
    expect(screen.getByText(problem)).toBeInTheDocument();
    unmount();

    // Started, the step cannot run either: its play control and the run to the next stop,
    // which would begin there, are blocked, each saying why.
    route({
      "GET /studies/providers": one,
      "GET /studies/s1": study({ steps }),
      "GET /studies/s1/chat": { summary: "", turns: [] },
    });
    render(<StudyPage studyId="s1" />, { wrapper: Wrapper });

    const play = await screen.findByRole("button", { name: "Run step 1" });
    expect(play).toBeDisabled();
    expect(play).toHaveAccessibleDescription(problem);
    const runOn = screen.getByRole("button", { name: "Run to the next stop" });
    expect(runOn).toBeDisabled();
    expect(runOn).toHaveAccessibleDescription(/Step 1 has a problem, shown on its card/);
  });

  it("shows the steps with a tool select where there are alternatives", async () => {
    const fetch = step({
      tool_id: "fetch_pdb",
      name: "Fetch from PDB",
      params: { pdb_id: "1CA2" },
      alternatives: [{ tool_id: "fetch_alphafold", name: "Fetch from AlphaFold" }],
    });
    const pockets = step({ id: "st2", position: 1, tool_id: "find_pockets", name: "Find pockets" });
    const planned = draft({ steps: [fetch, pockets] });
    const started = study({ steps: [fetch, pockets] });
    let current = planned;
    route({
      "GET /studies/providers": one,
      "GET /studies/s1": () => current,
      "GET /studies/s1/chat": { summary: "", turns: [] },
      "POST /studies/s1/steps/st1/tool": () => current,
      "POST /studies/s1/start": () => {
        current = started;
        return current;
      },
      "POST /studies/s1/run": () => {
        current = { ...started, running: true };
        return current;
      },
    });
    render(<StudyPage studyId="s1" />, { wrapper: Wrapper });

    const tool = await screen.findByRole("combobox", { name: "Tool for step 1" });
    expect(tool).toHaveValue("fetch_pdb");
    expect(screen.queryByRole("combobox", { name: "Tool for step 2" })).toBeNull();
    expect(screen.getByText("pdb_id: 1CA2")).toBeInTheDocument();
    // Nothing runs while it is a draft.
    expect(screen.queryByRole("button", { name: /^Run step/ })).toBeNull();

    fireEvent.change(tool, { target: { value: "fetch_alphafold" } });

    await waitFor(() => expect(requests("POST", "/studies/s1/steps/st1/tool")).toHaveLength(1));
    expect(requests("POST", "/studies/s1/steps/st1/tool")[0].data).toEqual({
      tool_id: "fetch_alphafold",
    });

    // Started, the steps can run, one at a time or on to the next stop.
    fireEvent.click(screen.getByRole("button", { name: "Start this study" }));
    fireEvent.click(await screen.findByRole("button", { name: "Run to the next stop" }));
    await waitFor(() => expect(requests("POST", "/studies/s1/run")).toHaveLength(1));
    expect(requests("POST", "/studies/s1/start")).toHaveLength(1);
    expect(requests("POST", "/studies/s1/run")[0].data).toEqual({});
    expect(screen.getByRole("button", { name: "Run step 2" })).toBeInTheDocument();

    // A run is said once, under the steps heading, and each control it blocks points there.
    expect(await screen.findAllByText(RUN_UNDER_WAY)).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Send" })).toHaveAccessibleDescription(RUN_UNDER_WAY);
    expect(screen.getByLabelText("Add a file")).toHaveAccessibleDescription(RUN_UNDER_WAY);
  });

  it("reads the thread again when the study moves", async () => {
    // A run writes its turns while no request of ours is open: the study's poll is the only
    // sign, so the thread follows the study's updated_at.
    const served = study();
    route({
      "GET /studies/providers": one,
      "GET /studies/s1": served,
      "GET /studies/s1/chat": { summary: "", turns: [] },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <StudyPage studyId="s1" />
      </QueryClientProvider>,
    );
    await screen.findByText(served.restated_goal);
    await waitFor(() => expect(requests("GET", "/studies/s1/chat")).toHaveLength(1));

    act(() => {
      client.setQueryData(["studies", "s1"], { ...served, updated_at: "2026-10-22T00:01:00Z" });
    });

    await waitFor(() => expect(requests("GET", "/studies/s1/chat")).toHaveLength(2));
  });
});
