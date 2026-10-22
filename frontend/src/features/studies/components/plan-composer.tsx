"use client";

import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { Button } from "@/shared/components/ui/button";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Textarea } from "@/shared/components/ui/textarea";
import { ApiError } from "@/shared/lib/api/custom-instance";
import { useBreadcrumbOverride } from "@/shared/lib/stores/breadcrumb-store";
import { AlertTriangle, CircleDot, Paperclip, PauseCircle, X } from "lucide-react";
import { useState } from "react";
import { useAskFirst, useDraftPlan, useProviders } from "../hooks/use-plans";
import { useStartStudy, useStudy } from "../hooks/use-studies";
import type { Plan, Question, Questions, Study } from "../types";
import { PROVIDER_LABELS, groupSteps, refinements } from "../types";
import { PlanTerms } from "./plan-terms";
import { QuestionCard } from "./question-card";
import { StudySteps } from "./study-steps";

/**
 * Say what you want to find out; read back a plan before anything runs.
 *
 * The plan is a proposal and the page is built to be argued with: it leads with
 * what the planner did not understand, says what would count as a result before
 * any result exists, and shows anything that would stop the plan executing as
 * written. Nothing here runs a step.
 *
 * It is also the study's own page. With `studyId` it opens on the plan the study
 * was started from, as it stood when started, so a study reads the same whether
 * it was just started here or opened from its address later. Nothing is asked
 * or planned in that mode.
 */
export function PlanComposer({ studyId }: { studyId?: string } = {}) {
  const providers = useProviders();
  const ask = useAskFirst();
  const draft = useDraftPlan();
  const start = useStartStudy();
  const saved = useStudy(studyId);
  const [goal, setGoal] = useState("");
  // Once started, the study lives under the plan on this same page. The form
  // above it is kept as the record of how the plan came about, and locked.
  const [startedId, setStartedId] = useState<string | null>(studyId ?? null);
  const started = startedId !== null;
  const [provider, setProvider] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  // Keyed by the question text, which is what travels back to the model.
  const [answers, setAnswers] = useState<Record<string, string>>({});
  // A saved study fills the form once, when it arrives; polling it afterwards
  // (while a step runs) must not overwrite anything.
  const [hydrated, setHydrated] = useState(false);
  if (saved.data && !hydrated) {
    setHydrated(true);
    setGoal(saved.data.goal);
    setAnswers(Object.fromEntries(saved.data.answers.map((a) => [a.ask, a.answer])));
  }
  const kept = keptPlanning(saved.data);
  // The id segment is the study's once there is one, and "new" until then. A
  // goal is a sentence and the breadcrumb bar is one line: a long label forces
  // the whole page wider than the window and it scrolls sideways.
  useBreadcrumbOverride(
    startedId ?? "new",
    startedId ? (goal.trim() ? crumbLabel(goal.trim()) : undefined) : "New study",
  );

  const available = providers.data ?? [];
  const nothingConfigured = providers.isSuccess && available.length === 0;
  const working = ask.isPending || draft.isPending;
  const asked = ask.data ?? kept.asked;
  // A question the planner could default arrives as an assumption and counts
  // as answered: leaving it alone is an answer. Shown after the ones it could
  // not, which are the reason this round exists.
  const asking = asked
    ? [
        ...asked.questions.filter((q) => q.default == null),
        ...asked.questions.filter((q) => q.default != null),
      ]
    : [];
  const unanswered = asking.filter((q) => !(answers[q.ask] ?? q.default ?? "").trim()).length;

  // The house convention: a blocked button says why, in a sentence, beside
  // itself, rather than sitting disabled with no explanation.
  const reason = nothingConfigured
    ? "No model provider is configured on this server."
    : !goal.trim()
      ? "Say what you want to find out."
      : goal.trim().length < 15
        ? "Give a little more detail, so the plan can be specific."
        : null;

  const failure =
    draft.error instanceof ApiError
      ? draft.error
      : ask.error instanceof ApiError
        ? ask.error
        : null;
  const plan = draft.data ?? kept.plan;
  // Everything the plan already handles, each carrying what it will do if
  // ignored. The plan on screen is complete without any of them being touched.
  const open = plan ? refinements(plan.questions) : [];
  // A plan-time question with no default means the planner could not pick
  // either way. It should be rare, and dropping it silently would be the
  // worst of the options, so it is shown first and said to be unsettled.
  const unsettled = plan ? plan.questions.filter((q) => q.default == null) : [];
  // Only offer to redraw when something actually differs from what the plan
  // already assumed. Otherwise the button invites a minute of waiting to be
  // told the same thing again.
  const changed =
    unsettled.some((question) => (answers[question.ask] ?? "").trim() !== "") ||
    open.some((question) => (answers[question.ask] ?? question.default ?? "") !== question.default);

  function shared() {
    return { goal: goal.trim(), provider: provider || undefined, file: file ?? undefined };
  }

  /** First press: find out what it needs to know, and plan nothing yet. */
  function submit() {
    if (reason || working) return;
    draft.reset();
    setAnswers({});
    ask.mutate(shared());
  }

  /**
   * Re-plan with the refinements as well. The whole plan is redrawn rather than
   * patched: there is no partial-update path, and a plan half-written against
   * old answers would be worse than a slower honest one.
   */
  function refine() {
    if (!plan || working) return;
    draft.mutate({
      ...shared(),
      answers: [...(asked?.questions ?? []), ...plan.questions].map((q) => ({
        ask: q.ask,
        answer: answers[q.ask] ?? q.default ?? "",
      })),
    });
  }

  /** Save the plan as a study, with the data it was planned against, and open it. */
  function startStudy() {
    if (!plan || plan.problems.length > 0 || start.isPending) return;
    start.mutate(
      {
        plan: {
          goal: plan.goal,
          restated_goal: plan.restated_goal,
          checks: plan.checks,
          stop_conditions: plan.stop_conditions,
          // Every question asked, either round, as it stood: what the scientist
          // said, or what the planner assumed and was left alone.
          answers: [...(asked?.questions ?? []), ...plan.questions].map((q) => ({
            ask: q.ask,
            answer: answers[q.ask] ?? q.default ?? "",
          })),
          // The plan itself, kept with the study so its page can show it again.
          planning: { asked, plan },
          steps: plan.steps.map((s) => ({
            tool_id: s.tool_id,
            purpose: s.purpose,
            params: s.params,
            gated: s.gated,
            always: s.always,
            expanded_from: s.expanded_from,
          })),
        },
        file: file ?? undefined,
      },
      {
        onSuccess: (study) => {
          setStartedId(study.id);
          // The address becomes the study's, without leaving the page: a
          // refresh or a bookmark finds the study, and nothing remounts.
          window.history.replaceState(window.history.state, "", `/studies/${study.id}`);
        },
      },
    );
  }

  /** Second press: plan, now against what the scientist actually said. */
  function planIt() {
    if (!asked || working) return;
    draft.mutate({
      ...shared(),
      answers: asked.questions.map((q) => ({
        ask: q.ask,
        answer: answers[q.ask] ?? q.default ?? "",
      })),
    });
  }

  function card(question: Question) {
    return (
      <QuestionCard
        key={question.ask}
        question={question}
        answer={answers[question.ask] ?? question.default ?? ""}
        disabled={working || started}
        onAnswer={(value) => setAnswers((previous) => ({ ...previous, [question.ask]: value }))}
      />
    );
  }

  if (studyId && saved.isLoadingError) {
    return (
      <QueryError
        title="Could not load this study"
        retry={saved.refetch}
        retrying={saved.isRefetching}
      />
    );
  }
  if (studyId && !hydrated) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  // Inherited, so one declaration covers every piece of prose the model writes
  // below -- restated goals, checks, stop conditions, problems. All of it can
  // quote back what the scientist pasted, and a sequence or an accession is one
  // token that will not wrap at spaces.
  return (
    <div className="w-full min-w-0 space-y-6 wrap-anywhere">
      <PageHeader
        title="Plan an analysis"
        description="Say what you want to find out. You get a plan to read and argue with, not a result."
      />

      <div className="space-y-3 rounded-sm border p-4">
        <Label htmlFor="goal">What do you want to find out?</Label>
        <Textarea
          id="goal"
          value={goal}
          rows={3}
          disabled={working || started}
          // The shared textarea is `field-sizing-content`, so it grows to fit
          // what it holds -- and a goal here can hold a pasted sequence, which
          // is one 270-character word. `wrap-anywhere` rather than
          // `break-words`: only `anywhere` lowers the min-content width, and
          // without that the column is forced past its share of the shell and
          // the whole page scrolls sideways. Measured, not assumed.
          className="wrap-anywhere"
          placeholder="I've got 48 compounds from our last two rounds against the same target. Help me work out what's driving activity and what to make next."
          onChange={(event) => setGoal(event.target.value)}
        />

        <div className="flex flex-wrap items-center gap-2">
          {/* Optional, and worth it: without the data the planner asks what it
              could have read, and a question answerable from the first ten rows
              is homework rather than a decision. Only the shape is sent -- counts,
              ranges and the values that do not fit. Structures are described, not
              quoted. */}
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-sm border px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted/50">
            <Paperclip className="size-3.5" aria-hidden="true" />
            {file ? file.name : "Add the data (optional)"}
            <input
              type="file"
              accept=".csv,.tsv,.tab,text/csv"
              className="sr-only"
              disabled={working || started}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          {file && (
            <Button
              variant="ghost"
              size="sm"
              aria-label="Remove the file"
              disabled={working || started}
              onClick={() => setFile(null)}
            >
              <X className="size-3.5" aria-hidden="true" />
            </Button>
          )}
          {available.length > 1 && (
            <Select value={provider} onValueChange={setProvider}>
              <SelectTrigger aria-label="Model provider" className="w-56">
                <SelectValue placeholder="Default model" />
              </SelectTrigger>
              <SelectContent>
                {available.map((option) => (
                  <SelectItem key={option.provider} value={option.provider}>
                    {PROVIDER_LABELS[option.provider] ?? option.provider} · {option.model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Button
            className="sm:ml-auto"
            disabled={Boolean(reason) || working || started}
            aria-describedby={reason ? "plan-reason" : undefined}
            onClick={submit}
          >
            {ask.isPending ? "Reading\u2026" : asked ? "Start again" : "Draft a plan"}
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">
          {file
            ? "Only the shape of the file is sent: how many rows, what each column holds, and the values that do not fit. Structures are described rather than quoted."
            : "Without the data, the questions come back as homework rather than decisions."}
        </p>

        {reason && !working && !started && (
          <p id="plan-reason" className="text-xs text-muted-foreground" aria-live="polite">
            {reason}
            {nothingConfigured && " Set STUDIO_ANTHROPIC_API_KEY or STUDIO_OPENAI_API_KEY."}
          </p>
        )}
      </div>

      {failure && (
        <div className="rounded-sm border border-destructive/40 bg-destructive/5 p-4" role="alert">
          <p className="text-sm font-medium">{failure.message}</p>
          {typeof failure.body === "object" &&
            failure.body !== null &&
            "detail" in failure.body &&
            typeof failure.body.detail === "string" && (
              <p className="mt-1 text-sm text-muted-foreground">{failure.body.detail}</p>
            )}
        </div>
      )}

      {asked && !plan && (
        <div className="space-y-5">
          <section className="space-y-1">
            <h2 className="text-sm font-medium">Daikon Studio&apos;s Understanding</h2>
            <p className="text-sm text-muted-foreground">{asked.restated_goal}</p>
          </section>

          <section className="space-y-3">
            <div>
              <h2 className="text-sm font-medium">Help Daikon Studio Understand More</h2>
              <p className="text-xs text-muted-foreground">
                Nothing has been planned yet. Answer what you can, skip what you cannot, and say
                when you are not sure. One with an answer already filled in is what Daikon Studio
                will assume; change it only if you disagree. Anything left open has to be settled by
                a step or named as an assumption rather than quietly decided.
              </p>
            </div>

            <ul className="space-y-2">{asking.map(card)}</ul>

            <div className="sticky bottom-0 z-10 flex flex-wrap items-center gap-3 rounded-lg border bg-background/95 p-3 backdrop-blur">
              <p className="text-xs text-muted-foreground" aria-live="polite">
                {unanswered === 0
                  ? "All answered."
                  : `${unanswered} of ${asked.questions.length} unanswered. That is allowed; the plan will say what it assumed.`}
              </p>
              <Button className="sm:ml-auto" disabled={working} onClick={planIt}>
                {draft.isPending ? "Planning\u2026" : "Draft the plan"}
              </Button>
            </div>
          </section>
        </div>
      )}

      {plan && (
        <div className="space-y-5">
          <section className="space-y-1">
            <h2 className="text-sm font-medium">Daikon Studio&apos;s Understanding</h2>
            <p className="text-sm text-muted-foreground">{plan.restated_goal}</p>
          </section>

          {/* Once started, the first round is part of the record too: without it a
              study whose questions were all settled before planning shows none. */}
          {started && asking.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-sm font-medium">What you answered</h2>
              <ul className="space-y-2">{asking.map(card)}</ul>
            </section>
          )}

          {(open.length > 0 || unsettled.length > 0) && (
            <section className="space-y-3">
              <div>
                <h2 className="text-sm font-medium">
                  {unsettled.length > 0 ? "Worth settling" : "Change any of these"}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {unsettled.length > 0
                    ? "The first of these has no answer the planner could defend either way, so the plan below may be wrong until you say. The rest it already handles; change one only if you disagree."
                    : "The plan below already does what each of these says. Change one only if you disagree; leaving them alone is a complete answer."}
                  {plan.questions_left > 0
                    ? ` Up to ${plan.questions_left} more question${plan.questions_left === 1 ? "" : "s"} may be raised after that.`
                    : " No more questions will be raised."}
                </p>
              </div>

              <ul className="space-y-2">{[...unsettled, ...open].map(card)}</ul>

              {!started && (
                <div className="flex flex-wrap items-center gap-3">
                  <p className="text-xs text-muted-foreground" aria-live="polite">
                    {changed
                      ? "Updating redraws the plan from everything you have said."
                      : "Nothing changed yet."}
                  </p>
                  <Button
                    variant="outline"
                    className="sm:ml-auto"
                    disabled={working || !changed}
                    onClick={refine}
                  >
                    {draft.isPending ? "Updating\u2026" : "Update the plan"}
                  </Button>
                </div>
              )}
            </section>
          )}

          {plan.problems.length > 0 && (
            <section
              className="space-y-2 rounded-sm border border-destructive/40 bg-destructive/5 p-4"
              role="alert"
            >
              <h2 className="flex items-center gap-2 text-sm font-medium">
                <AlertTriangle className="size-4" aria-hidden="true" />
                This plan could not run as written
              </h2>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {plan.problems.map((problem) => (
                  <li key={problem.message}>{problem.message}</li>
                ))}
              </ul>
            </section>
          )}

          {startedId ? (
            <StudySteps studyId={startedId} />
          ) : (
            <section className="space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-sm font-medium">The plan</h2>
                <Button
                  size="sm"
                  className="ml-auto"
                  disabled={plan.problems.length > 0 || working || start.isPending}
                  aria-describedby={plan.problems.length > 0 ? "start-reason" : undefined}
                  onClick={startStudy}
                >
                  {start.isPending ? "Starting…" : "Start this study"}
                </Button>
              </div>
              {plan.problems.length > 0 && (
                <p id="start-reason" className="text-xs text-muted-foreground">
                  Fix the problems above before starting: a study runs the plan as written.
                </p>
              )}
              {start.error && (
                <p className="text-sm text-destructive" role="alert">
                  {start.error.message}
                </p>
              )}
              <ol className="space-y-2">
                {groupSteps(plan.steps).map((group) => (
                  <li
                    key={`${group.from ?? ""}-${group.steps[0].tool_id}`}
                    className="rounded-sm border p-3"
                  >
                    {group.from && (
                      <>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          One step: {group.from.replace(/_/g, " ")}
                        </p>
                        {/* The planner gave one purpose for the whole sequence, so
                          it belongs to the group. Repeating it on each part is
                          how a grouped plan reads as three times the work. */}
                        <p className="mb-2 mt-1 text-xs text-muted-foreground">
                          {group.steps[0].purpose}
                        </p>
                      </>
                    )}
                    {group.steps.map((step) => (
                      <div key={step.tool_id} className="flex items-start gap-2 py-1">
                        <CircleDot
                          className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
                          aria-hidden="true"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium">{step.name}</p>
                          {!group.from && (
                            <p className="text-xs text-muted-foreground">{step.purpose}</p>
                          )}
                        </div>
                        {step.gated && (
                          <span className="flex items-center gap-1 whitespace-nowrap text-xs text-muted-foreground">
                            <PauseCircle className="size-3.5" aria-hidden="true" />
                            stops for you
                          </span>
                        )}
                      </div>
                    ))}
                  </li>
                ))}
              </ol>
            </section>
          )}

          <PlanTerms checks={plan.checks} stopConditions={plan.stop_conditions} />

          {plan.skipped.length > 0 && (
            <section className="space-y-2">
              <h2 className="text-sm font-medium">Considered and left out</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                {plan.skipped.map((item) => (
                  <li key={item.message}>{item.message}</li>
                ))}
              </ul>
            </section>
          )}

          <p className="border-t pt-3 text-xs text-muted-foreground">
            Written by {plan.written_by}. A plan is a judgement, so it is worth knowing what made
            it. Nothing has run.
          </p>
        </div>
      )}

      {startedId && !plan && (
        <div className="space-y-5">
          <p className="text-xs text-muted-foreground">
            The plan this study was started from was not kept: it was started before plans were
            saved with their studies.
          </p>
          <StudySteps studyId={startedId} />
        </div>
      )}
    </div>
  );
}

const CRUMB_CHARS = 60;

function crumbLabel(goal: string): string {
  return goal.length > CRUMB_CHARS ? `${goal.slice(0, CRUMB_CHARS - 1).trimEnd()}…` : goal;
}

/** The plan a study was started from, or nothing for one started before plans were kept. */
function keptPlanning(study: Study | undefined): { asked?: Questions; plan?: Plan } {
  const planning = study?.planning;
  if (!planning || typeof planning.plan !== "object" || planning.plan === null) return {};
  return planning as { asked?: Questions; plan?: Plan };
}
