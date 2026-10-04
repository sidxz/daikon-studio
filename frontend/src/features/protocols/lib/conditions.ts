import type { Condition } from "@/features/engines";

type Scoped = Partial<Pick<Condition, "key" | "tasks">>;

/**
 * Whether a setting means anything for a dataset with these tasks. A setting that
 * names tasks (class weighting, say) applies only when the dataset has one of them;
 * unknown tasks, as before a dataset is chosen, hide nothing.
 */
export function appliesToTasks(condition: Scoped, tasks: string[] | undefined): boolean {
  return !condition.tasks?.length || !tasks || condition.tasks.some((task) => tasks.includes(task));
}

type Bounded = Pick<Condition, "type" | "minimum" | "maximum">;

/**
 * Why the worker would refuse this value, in the words the form shows under the
 * field, or null when it is acceptable. The same bounds and whole-number check as
 * the backend's `validate_conditions`: without it the form submits a run that can
 * only fail. An empty field is acceptable, since the default then applies.
 */
export function conditionError(condition: Bounded, value: unknown): string | null {
  if (condition.type !== "integer" && condition.type !== "number") return null;
  if (value === undefined || value === null || value === "") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return "Enter a number.";
  if (condition.type === "integer" && !Number.isInteger(number)) return "Enter a whole number.";
  const { minimum, maximum } = condition;
  if (minimum != null && maximum != null && (number < minimum || number > maximum)) {
    return `Must be between ${minimum} and ${maximum}.`;
  }
  if (minimum != null && number < minimum) return `Must be at least ${minimum}.`;
  if (maximum != null && number > maximum) return `Must be at most ${maximum}.`;
  return null;
}

/** Whether every setting the form shows holds a value the worker will accept. */
export function conditionsValid(
  conditions: (Bounded & Scoped)[],
  values: Record<string, unknown>,
  tasks: string[] | undefined,
  pinned?: Record<string, unknown>,
): boolean {
  return conditions.every(
    (condition) =>
      !appliesToTasks(condition, tasks) ||
      (pinned != null && condition.key != null && condition.key in pinned) ||
      conditionError(condition, values[condition.key ?? ""]) === null,
  );
}

/**
 * `values` without the settings the form hides for this dataset. A value typed while
 * a different dataset was chosen would otherwise ride along, unseen, in the request.
 */
export function withoutInapplicable(
  specs: Scoped[],
  values: Record<string, unknown>,
  tasks: string[] | undefined,
): Record<string, unknown> {
  const hidden = new Set(specs.filter((spec) => !appliesToTasks(spec, tasks)).map((s) => s.key));
  return Object.fromEntries(Object.entries(values).filter(([key]) => !hidden.has(key)));
}
