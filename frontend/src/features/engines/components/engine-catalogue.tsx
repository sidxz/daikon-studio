"use client";

import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { Search } from "lucide-react";
import { useState } from "react";
import { useEngines } from "../hooks/use-engines";
import { TASK_LABELS } from "../types";
import { EngineCard, computeLabel } from "./engine-card";

const CATEGORIES = [
  { value: "all", label: "All engines" },
  { value: "molecule", label: "Small molecules" },
  { value: "sequence", label: "Proteins" },
] as const;

export function EngineCatalogue() {
  const { data: engines, isLoading, isError, refetch, isFetching } = useEngines();
  const [category, setCategory] = useState("all");
  const [search, setSearch] = useState("");
  const [task, setTask] = useState("all");
  const [lane, setLane] = useState("all");
  const hasFilters = search !== "" || task !== "all" || lane !== "all";
  const tasks = [...new Set(engines?.flatMap((engine) => engine.tasks) ?? [])];
  const lanes = [...new Set(engines?.map((engine) => engine.lane) ?? [])];
  const query = search.trim().toLowerCase();
  const visible = engines?.filter(
    (engine) =>
      (category === "all" || engine.structure_kinds.includes(category)) &&
      (!query || `${engine.name} ${engine.id}`.toLowerCase().includes(query)) &&
      (task === "all" || engine.tasks.includes(task)) &&
      (lane === "all" || engine.lane === lane),
  );

  function clearFilters() {
    setSearch("");
    setTask("all");
    setLane("all");
  }

  return (
    <div className="w-full min-w-0 space-y-6">
      <PageHeader
        title="Engines"
        description="Models for molecules and proteins, with supported tasks and training settings."
      />

      {isLoading && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" aria-label="Loading engines">
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      )}

      {isError && (
        <QueryError title="Could not load engines" retry={() => refetch()} retrying={isFetching} />
      )}

      {engines && (
        <Tabs value={category} onValueChange={setCategory}>
          <TabsList aria-label="Engine input type">
            {CATEGORIES.map(({ value, label }) => (
              <TabsTrigger key={value} value={value}>
                {label}
                <span className="rounded bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground">
                  {value === "all"
                    ? engines.length
                    : engines.filter((engine) => engine.structure_kinds.includes(value)).length}
                </span>
              </TabsTrigger>
            ))}
          </TabsList>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative w-full sm:mr-auto sm:w-72">
              <Search
                className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                aria-label="Search engines"
                placeholder="Search engines…"
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={task} onValueChange={setTask}>
              <SelectTrigger aria-label="Filter by task" className="flex-1 sm:w-48 sm:flex-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All tasks</SelectItem>
                {tasks.map((value) => (
                  <SelectItem key={value} value={value}>
                    {TASK_LABELS[value] ?? value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={lane} onValueChange={setLane}>
              <SelectTrigger aria-label="Filter by compute" className="flex-1 sm:w-40 sm:flex-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All compute</SelectItem>
                {lanes.map((value) => (
                  <SelectItem key={value} value={value}>
                    {computeLabel(value)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {hasFilters && (
              <Button variant="ghost" size="sm" onClick={clearFilters}>
                Clear filters
              </Button>
            )}
          </div>

          {CATEGORIES.map(({ value }) => (
            <TabsContent key={value} value={value} className="space-y-3">
              <output className="block text-xs text-muted-foreground">
                {visible?.length} {visible?.length === 1 ? "engine" : "engines"}
                {hasFilters ? " matching your filters" : " available"}
              </output>
              {visible?.length ? (
                <div className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {visible.map((engine) => (
                    <EngineCard key={engine.id} engine={engine} />
                  ))}
                </div>
              ) : (
                <div className="rounded-sm border border-dashed px-6 py-16 text-center">
                  <h2 className="text-sm font-medium">
                    {hasFilters ? "No engines match your filters" : "No engines available"}
                  </h2>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {hasFilters
                      ? "Try another search or clear your filters to see more engines."
                      : "Engines will appear here when they are added to the catalogue."}
                  </p>
                  {hasFilters && (
                    <Button variant="outline" className="mt-4" onClick={clearFilters}>
                      Clear filters
                    </Button>
                  )}
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>
      )}
    </div>
  );
}
