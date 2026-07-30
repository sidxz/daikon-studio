import { navigation } from "@/shared/lib/navigation";
import Link from "next/link";

/**
 * Placeholder home. A real dashboard needs numbers worth glancing at, and
 * nothing in Phase 1's API aggregates any -- so this is honest signposting
 * rather than invented density.
 */
export default function DashboardPage() {
  const sections = navigation.filter((group) => group.label !== "Overview");

  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 p-2">
      <div>
        <h1 className="text-lg font-semibold">DAIKON Studio</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload data, train a protocol, read its scorecard, then run it across a compound set and
          triage what comes back.
        </p>
      </div>

      {sections.map((group) => (
        <div key={group.label}>
          <h2 className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
            {group.label}
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {group.items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex h-full items-center gap-3 rounded-lg border border-border p-4 transition-colors hover:bg-muted/50"
              >
                <item.icon className="size-5 shrink-0 text-muted-foreground" />
                <span className="text-sm font-medium">{item.title}</span>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
