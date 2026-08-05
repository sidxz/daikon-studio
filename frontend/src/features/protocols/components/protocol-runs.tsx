"use client";

import { Badge } from "@/shared/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import Link from "next/link";
import { useProtocolRuns } from "../hooks/use-protocols";

const STATUS_TONE: Record<string, string> = {
  ready: "text-success",
  failed: "text-destructive",
  cancelled: "text-muted-foreground",
};

/**
 * Everything this Protocol has been used for: the training Run that produced
 * it, and every prediction Run since.
 *
 * The training Run is the provenance of the numbers on this page; the
 * prediction Runs are what the Protocol has actually been asked, which is the
 * only record of whether a published model is in use or was published and
 * forgotten.
 */
export function ProtocolRuns({ protocolId }: { protocolId: string }) {
  const { data, isLoading, isError } = useProtocolRuns(protocolId);

  if (isLoading) return <Skeleton className="h-32 w-full" />;
  if (isError) return null;

  const runs = data?.items ?? [];
  if (runs.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Run history</CardTitle>
        <p className="text-sm text-muted-foreground">
          The training run behind this Protocol, and every prediction made with it.
        </p>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="pb-2 pr-4 font-medium">Kind</th>
              <th className="pb-2 pr-4 font-medium">Status</th>
              <th className="pb-2 pr-4 font-medium">Started</th>
              <th className="pb-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id} className="border-b last:border-0">
                <td className="py-2 pr-4">
                  <Badge variant="outline" className="font-normal">
                    {run.kind}
                  </Badge>
                </td>
                <td className={`py-2 pr-4 ${STATUS_TONE[run.status] ?? ""}`}>
                  {run.status}
                  {run.status === "running" && ` · ${Math.round(run.progress * 100)}%`}
                </td>
                <td className="py-2 pr-4 text-muted-foreground">
                  {new Date(run.created_at).toLocaleString()}
                </td>
                <td className="py-2 text-right">
                  <Link
                    href={`/runs/${run.id}`}
                    className="text-xs text-muted-foreground underline-offset-4 hover:underline"
                  >
                    View
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
