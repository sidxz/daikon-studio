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
