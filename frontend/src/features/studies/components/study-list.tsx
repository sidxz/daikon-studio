"use client";

import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { StatusDot, type StatusTone } from "@/shared/components/status-dot";
import { buttonVariants } from "@/shared/components/ui/button";
import { EmptyState } from "@/shared/components/ui/empty-state";
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { ListChecks } from "lucide-react";
import Link from "next/link";
import { useStudies } from "../hooks/use-studies";
import type { Study } from "../types";

function look(study: Study): { tone: StatusTone; word: string } {
  if (study.running) return { tone: "active", word: "Running" };
  if (study.status === "done") return { tone: "success", word: "Done" };
  if (study.status === "failed") return { tone: "failed", word: "Failed" };
  return { tone: "muted", word: "Waiting for you" };
}

export function StudyList() {
  const { data, isLoadingError, refetch, isRefetching } = useStudies();
  const plan = (
    <Link href="/studies/new" className={buttonVariants()}>
      Plan a study
    </Link>
  );

  return (
    <div className="w-full min-w-0 space-y-6">
      <PageHeader
        title="Studies"
        description="Plans you started. Each runs one step at a time and keeps what every step found."
        action={plan}
      />
      {isLoadingError ? (
        <QueryError title="Could not load your studies" retry={refetch} retrying={isRefetching} />
      ) : !data ? (
        <Skeleton className="h-48 w-full" />
      ) : data.length === 0 ? (
        <EmptyState
          icon={ListChecks}
          title="No studies yet"
          description="Draft a plan, read it, and start it from the plan page."
          action={plan}
        />
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Goal</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Steps done</TableHead>
              <TableHead>Started</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((study) => {
              const { tone, word } = look(study);
              const done = study.steps.filter((s) => s.status === "done").length;
              return (
                <TableRow key={study.id}>
                  <TableCell className="max-w-md whitespace-normal">
                    <Link href={`/studies/${study.id}`} className="hover:underline">
                      {study.goal}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <span className="flex items-center gap-2">
                      <StatusDot tone={tone} />
                      {word}
                    </span>
                  </TableCell>
                  <TableCell>
                    {done} of {study.steps.length}
                  </TableCell>
                  <TableCell>{new Date(study.created_at).toLocaleDateString()}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
