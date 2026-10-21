import type {
  PlanResponse,
  ProviderResponse,
  QuestionResponse,
  QuestionsResponse,
  StepResponse,
} from "@/shared/lib/api/model";

// Generated DTOs, re-exported under domain names. Never redeclare their shape --
// `pnpm generate:api` is what keeps them true.
export type Plan = PlanResponse;
export type PlanStep = StepResponse;
export type Provider = ProviderResponse;
export type Question = QuestionResponse;
export type Questions = QuestionsResponse;

export const PROVIDER_LABELS: Record<string, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
};

/**
 * Steps grouped by the composite they came from, preserving order.
 *
 * The planner chose "build the SAR table"; the plan records the two steps that
 * will actually run. Showing the group is how the page reads the way the
 * planner thought, without hiding what gets recorded.
 */
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
 * Questions the plan already handles, each with what it will do if ignored.
 * Separate from the blocking ones because the plan on screen is complete: these
 * change a detail, and answering none of them is a valid outcome.
 */
export function refinements(questions: Question[]): Question[] {
  return questions.filter((question) => question.default != null);
}

export function groupSteps(steps: PlanStep[]): { from: string | null; steps: PlanStep[] }[] {
  const groups: { from: string | null; steps: PlanStep[] }[] = [];
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
