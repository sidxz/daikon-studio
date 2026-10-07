import type { ReactNode } from "react";

export function ScorecardSection({
  title,
  description,
  children,
}: { title: string; description: string; children: ReactNode }) {
  return (
    <section aria-label={title} className="space-y-4 border-t pt-6">
      <header>
        <h3 className="text-base font-semibold">{title}</h3>
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      </header>
      <div className="space-y-4">{children}</div>
    </section>
  );
}
