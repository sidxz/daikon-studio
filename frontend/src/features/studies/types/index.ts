import type { StatusTone } from "@/shared/components/status-dot";
import type {
  ProposalResponse,
  ProviderResponse,
  QuestionResponse,
  StepStateResponse,
  StudyResponse,
  ThreadResponse,
  TurnResponse,
  ValueResponse,
  ValueSummaryResponse,
} from "@/shared/lib/api/model";

// Generated DTOs, re-exported under domain names. Never redeclare their shape.
export type Study = StudyResponse;
export type StudyStep = StepStateResponse;
export type ValueSummary = ValueSummaryResponse;
export type StudyValue = ValueResponse;
export type Thread = ThreadResponse;
export type Turn = TurnResponse;
export type StudyProposal = ProposalResponse;
export type Question = QuestionResponse;
export type Provider = ProviderResponse;

export const STEP_STATUS_COPY: Record<string, string> = {
  pending: "Not run yet",
  running: "Running",
  done: "Completed",
  failed: "Failed",
  skipped: "Skipped",
};

export const STEP_TONE: Record<string, StatusTone> = {
  pending: "muted",
  running: "active",
  done: "success",
  failed: "failed",
  skipped: "muted",
};

/**
 * Why a control that writes to the study is blocked while a run is in flight. The server
 * refuses every such write until the run stops. The page says so once, under the steps
 * heading (`STEPS_REASON_ID`), and each control it blocks points there.
 */
export const RUN_UNDER_WAY =
  "Execution is in progress. Changes are available when execution stops.";

/** The line under the steps heading that says why Start or a run is blocked. */
export const STEPS_REASON_ID = "steps-reason";

/**
 * The DOM id of a step's card, so a turn in the thread can scroll to the step it is about.
 * The thread sits beside the steps and talks about one at a time; without this the reader
 * matches "step 2" to a card by hand, in a column that scrolls on its own.
 */
export const stepAnchor = (stepId: string) => `step-${stepId}`;

export const PROVIDER_LABELS: Record<string, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
};

/**
 * Offered on every question, whatever the model proposed, and never produced by
 * it. A chemist often genuinely does not know whether the assay changed between
 * two runs, and that is a different answer from either yes or no: the plan has
 * to settle it with a step or name it as an assumption rather than pick. Without
 * this option people guess, and a guess recorded as an answer is worse than an
 * acknowledged unknown.
 */
export const NOT_SURE = "Not sure";
