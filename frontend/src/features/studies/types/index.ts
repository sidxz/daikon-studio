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
  done: "Done",
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

/**
 * Steps grouped by the composite they came from, preserving order.
 *
 * The planner chose "build the SAR table"; the plan records the two steps that
 * will actually run. Showing the group is how the page reads the way the
 * planner thought, without hiding what gets recorded.
 */
export function groupSteps(steps: StudyStep[]): { from: string | null; steps: StudyStep[] }[] {
  const groups: { from: string | null; steps: StudyStep[] }[] = [];
  for (const step of steps) {
    const last = groups.at(-1);
    if (last && last.from !== null && last.from === step.expanded_from) {
      last.steps.push(step);
    } else {
      groups.push({ from: step.expanded_from, steps: [step] });
    }
  }
  return groups;
}
