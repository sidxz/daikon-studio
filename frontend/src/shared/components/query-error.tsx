import { Button } from "@/shared/components/ui/button";
import { AlertCircle, RefreshCw } from "lucide-react";

export function QueryError({
  title,
  retry,
  retrying = false,
  description = "Your work is still here. Try loading this information again.",
}: {
  title: string;
  retry: () => unknown;
  retrying?: boolean;
  description?: string;
}) {
  return (
    <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/5 p-5">
      <div className="flex items-start gap-3">
        <AlertCircle aria-hidden className="mt-0.5 size-5 shrink-0 text-destructive" />
        <div className="min-w-0">
          <p className="text-sm font-semibold">{title}</p>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{description}</p>
          <Button variant="outline" size="sm" className="mt-3" onClick={retry} disabled={retrying}>
            <RefreshCw aria-hidden className={retrying ? "motion-safe:animate-spin" : undefined} />
            {retrying ? "Trying again…" : "Try again"}
          </Button>
        </div>
      </div>
    </div>
  );
}
