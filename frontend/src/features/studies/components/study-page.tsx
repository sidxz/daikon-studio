"use client";

import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { Button } from "@/shared/components/ui/button";
import { Label } from "@/shared/components/ui/label";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Textarea } from "@/shared/components/ui/textarea";
import { useBreadcrumbOverride } from "@/shared/lib/stores/breadcrumb-store";
import { cn } from "@/shared/lib/utils";
import { FastForward, Paperclip, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import {
  useAttachFiles,
  useOpenStudy,
  useProviders,
  useRunTo,
  useStartStudy,
  useStudy,
} from "../hooks/use-studies";
import type { Provider, Study } from "../types";
import { PROVIDER_LABELS, RUN_UNDER_WAY, STEPS_REASON_ID } from "../types";
import { PlanTerms } from "./plan-terms";
import { MESSAGE_ID, StudyChat } from "./study-chat";
import { StudySteps } from "./study-steps";
import { ValueBlock } from "./value-block";

// A goal shorter than this is not yet something to ask questions about; the server agrees.
const MIN_GOAL = 15;
const CRUMB_CHARS = 60;

/** A goal is a sentence and the breadcrumb bar is one line; a long label widens the page. */
function crumbLabel(goal: string): string {
  return goal.length > CRUMB_CHARS ? `${goal.slice(0, CRUMB_CHARS - 1).trimEnd()}…` : goal;
}

/**
 * A file picker drawn as a button. The input stays inside the label, so a keyboard reaches it
 * and the ring shows where focus is. Its value is cleared after every pick, so choosing the
 * same file again, after removing it or after a refusal, still counts as a choice.
 */
function FileButton({
  label,
  multiple = false,
  disabled = false,
  describedBy,
  onFiles,
}: {
  label: string;
  multiple?: boolean;
  disabled?: boolean;
  /** The id of the line saying why it is disabled, when that is said elsewhere. */
  describedBy?: string;
  onFiles: (files: File[]) => void;
}) {
  return (
    <label
      className={cn(
        "inline-flex cursor-pointer items-center gap-1.5 rounded-sm border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted/50 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <Paperclip className="size-3.5" aria-hidden="true" />
      {label}
      <input
        type="file"
        multiple={multiple}
        accept=".csv,.tsv,.tab,.pdb"
        className="sr-only"
        disabled={disabled}
        aria-describedby={describedBy}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length > 0) onFiles(files);
        }}
      />
    </label>
  );
}

/** Which model answers; offered only when the server can reach more than one. */
function ProviderSelect({
  choices,
  value,
  onChange,
}: {
  choices: Provider[];
  value: string;
  onChange: (provider: string) => void;
}) {
  return (
    <select
      aria-label="Model"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-9 max-w-full rounded-md border bg-background px-3 text-sm"
    >
      <option value="">Default model</option>
      {choices.map((choice) => (
        <option key={choice.provider} value={choice.provider}>
          {`${PROVIDER_LABELS[choice.provider] ?? choice.provider} · ${choice.model}`}
        </option>
      ))}
    </select>
  );
}

/** The first message: what to find out, the files to hand, and nothing planned yet. */
function OpenForm({
  provider,
  picker,
  unconfigured,
  onOpened,
}: {
  provider: string;
  picker: ReactNode;
  unconfigured: boolean;
  onOpened: (study: Study) => void;
}) {
  const open = useOpenStudy();
  const [goal, setGoal] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  useBreadcrumbOverride("new", "New study");

  const words = goal.trim();
  // The house rule: a blocked button says why, in a sentence beside itself.
  const reason = unconfigured
    ? "No model provider is configured on this server. Set STUDIO_ANTHROPIC_API_KEY or STUDIO_OPENAI_API_KEY."
    : !words
      ? "Say what you want to find out."
      : words.length < MIN_GOAL
        ? "Give a little more detail, so Daikon Studio can ask about the right things."
        : null;

  function submit() {
    if (reason || open.isPending) return;
    open.mutate(
      { message: words, provider: provider || undefined, files },
      { onSuccess: onOpened },
    );
  }

  // Inherited, so the goal and the refusal can quote a pasted sequence, which is one long word.
  return (
    <div className="w-full min-w-0 space-y-6 wrap-anywhere">
      <PageHeader
        title="New study"
        description="Say what you want to find out. Daikon Studio asks what it needs to know, then writes the plan with you. Nothing runs until you start the study."
      />

      <div className="space-y-3 rounded-sm border p-4">
        <Label htmlFor="goal">What do you want to find out?</Label>
        <Textarea
          id="goal"
          value={goal}
          rows={3}
          disabled={open.isPending}
          // `wrap-anywhere`, not `break-words`: only `anywhere` lowers the min-content width, and
          // a pasted sequence would otherwise push the column past its share of the page.
          className="wrap-anywhere"
          placeholder="I've got 48 compounds from our last two rounds against the same target. Help me work out what's driving activity and what to make next."
          onChange={(event) => setGoal(event.target.value)}
        />

        <div className="flex flex-wrap items-center gap-2">
          <FileButton
            label={
              files.length > 0 ? files.map((file) => file.name).join(", ") : "Add data (optional)"
            }
            multiple
            disabled={open.isPending}
            onFiles={setFiles}
          />
          {files.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              aria-label="Remove the files"
              disabled={open.isPending}
              onClick={() => setFiles([])}
            >
              <X className="size-3.5" aria-hidden="true" />
            </Button>
          )}
          {picker}
          <Button
            className="sm:ml-auto"
            disabled={Boolean(reason) || open.isPending}
            aria-describedby={reason ? "open-reason" : undefined}
            onClick={submit}
          >
            {open.isPending ? "Reading…" : "Ask Daikon Studio"}
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">
          At most two files: one compound table (.csv, .tsv or .tab) and one structure (.pdb). The
          model reads a summary of each and can ask to see up to 50 rows of a table, with structure
          and sequence columns left out.
        </p>
        {reason && (
          <p id="open-reason" className="text-xs text-muted-foreground" aria-live="polite">
            {reason}
          </p>
        )}
      </div>

      {open.error && (
        <div
          className="rounded-sm border border-destructive/40 bg-destructive/5 p-4 text-sm"
          role="alert"
        >
          {open.error.message}
        </div>
      )}
    </div>
  );
}

/** A study in any phase: the plan and its steps in the main column, the thread in the rail. */
function StudyView({
  studyId,
  provider,
  picker,
}: {
  studyId: string;
  provider: string;
  picker: ReactNode;
}) {
  const { data: study, isLoadingError, refetch, isRefetching } = useStudy(studyId);
  const start = useStartStudy(studyId);
  const runTo = useRunTo(studyId);
  const attach = useAttachFiles(studyId);
  const [about, setAbout] = useState<string | null>(null);
  useBreadcrumbOverride(studyId, study ? crumbLabel(study.goal) : undefined);

  if (isLoadingError) {
    return <QueryError title="Could not load this study" retry={refetch} retrying={isRefetching} />;
  }
  if (!study) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const draft = study.status === "draft";
  const problem = study.steps.find((step) => step.problem);
  // The house rule: a blocked button says why, in a sentence beside itself.
  const blocked = draft
    ? study.steps.length === 0
      ? "There is no plan to start yet."
      : problem
        ? `Step ${problem.position + 1} has a problem, shown on its card. Settle it before starting: a study runs the plan as written.`
        : null
    : study.running
      ? RUN_UNDER_WAY
      : study.status === "done"
        ? "Every step has run."
        : null;

  /** The step a message will be about, and the cursor in the box to write it. */
  function talk(stepId: string) {
    setAbout(stepId);
    document.getElementById(MESSAGE_ID)?.focus();
  }

  return (
    // No `items-start`: the rail has to be as tall as the steps beside it, or `sticky` has
    // nothing to stick within and the thread scrolls away from the result it is about.
    <div className="flex w-full min-w-0 flex-col gap-6 xl:flex-row">
      <div className="w-full min-w-0 space-y-6 wrap-anywhere">
        <PageHeader title={draft ? "Draft study" : "Study"} description={study.goal} />

        <section className="space-y-1">
          <h2 className="text-sm font-medium">Daikon Studio&apos;s understanding</h2>
          <p className="text-sm text-muted-foreground">
            {study.restated_goal || "Daikon Studio has not said how it reads the goal yet."}
          </p>
        </section>

        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-sm font-medium">What you brought</h2>
            <span className="ml-auto">
              <FileButton
                label={attach.isPending ? "Adding…" : "Add a file"}
                disabled={study.running || attach.isPending}
                describedBy={study.running ? STEPS_REASON_ID : undefined}
                onFiles={(files) => attach.mutate(files)}
              />
            </span>
          </div>
          {study.brought.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Nothing yet. A study takes one compound table (.csv, .tsv or .tab) and one structure
              (.pdb).
            </p>
          ) : (
            study.brought.map((value) => (
              <ValueBlock key={value.id} studyId={study.id} value={value} />
            ))
          )}
        </section>

        <section className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-sm font-medium">The steps</h2>
            {draft ? (
              <Button
                size="sm"
                className="ml-auto"
                disabled={Boolean(blocked) || start.isPending}
                aria-describedby={blocked ? STEPS_REASON_ID : undefined}
                onClick={() => start.mutate()}
              >
                {start.isPending ? "Starting…" : "Start this study"}
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                className="ml-auto"
                disabled={Boolean(blocked) || runTo.isPending}
                aria-describedby={blocked ? STEPS_REASON_ID : undefined}
                onClick={() => runTo.mutate({})}
              >
                <FastForward aria-hidden="true" />
                Run to the next stop
              </Button>
            )}
          </div>
          {blocked && (
            <p id={STEPS_REASON_ID} className="text-xs text-muted-foreground">
              {blocked}
            </p>
          )}
          {start.error && (
            <p className="text-sm text-destructive" role="alert">
              {start.error.message}
            </p>
          )}
          <StudySteps studyId={study.id} onTalk={talk} />
        </section>

        <PlanTerms
          checks={study.checks}
          stopConditions={study.stop_conditions}
          amendedAfterSteps={study.amended_after_steps}
        />
      </div>

      <aside className="w-full min-w-0 shrink-0 xl:w-[30rem]">
        {/* Pinned, so a claim can be read against the result it points at rather than from
            memory after scrolling. */}
        <div className="flex flex-col gap-3 xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)]">
          {picker}
          <StudyChat
            studyId={study.id}
            provider={provider}
            about={about}
            onClearAbout={() => setAbout(null)}
          />
        </div>
      </aside>
    </div>
  );
}

/**
 * A study from its first message. Without an id it is the opening form; posting it opens a
 * draft, and the page becomes that study's in place. With an id it is the study, in whatever
 * phase it is in: asked, planned, started or done.
 */
export function StudyPage({ studyId }: { studyId?: string } = {}) {
  const [id, setId] = useState(studyId ?? null);
  // The study keeps no provider, so the one chosen here goes with every message.
  const [provider, setProvider] = useState("");
  const providers = useProviders();
  const choices = providers.data ?? [];
  const picker =
    choices.length > 1 ? (
      <ProviderSelect choices={choices} value={provider} onChange={setProvider} />
    ) : null;

  if (id) return <StudyView studyId={id} provider={provider} picker={picker} />;
  return (
    <OpenForm
      provider={provider}
      picker={picker}
      unconfigured={providers.isSuccess && choices.length === 0}
      onOpened={(opened) => {
        setId(opened.id);
        // The address becomes the study's without leaving the page: a refresh or a bookmark
        // finds the study, and nothing remounts.
        window.history.replaceState(window.history.state, "", `/studies/${opened.id}`);
      }}
    />
  );
}
