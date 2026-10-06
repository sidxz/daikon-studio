import type { ReactNode } from "react";

/** One page title, with room for a readable description and a primary action. */
export function PageHeader({
  title,
  description,
  action,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 flex-[1_1_16rem] space-y-1.5">
        <h1 className="break-words text-2xl font-normal tracking-tight sm:text-[1.75rem]">
          {title}
        </h1>
        {description && (
          <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">{description}</p>
        )}
      </div>
      {action && (
        <div className="flex max-w-full shrink-0 flex-wrap items-center gap-2">{action}</div>
      )}
    </div>
  );
}
