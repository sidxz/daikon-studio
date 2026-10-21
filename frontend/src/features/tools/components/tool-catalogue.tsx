"use client";

import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/components/ui/tabs";
import { Search } from "lucide-react";
import { useState } from "react";
import { useTools } from "../hooks/use-tools";
import { AREAS } from "../types";
import { ToolCard } from "./tool-card";

/**
 * The catalogue, in the order a chemist performs these steps rather than
 * alphabetically, so the page reads as a piece of work rather than a list. The
 * backend owns that order; nothing here re-sorts it.
 */
export function ToolCatalogue() {
  const { data: tools, isLoading, isError, refetch, isFetching } = useTools();
  const [search, setSearch] = useState("");
  const [decisionsOnly, setDecisionsOnly] = useState(false);
  const [area, setArea] = useState<string>("all");

  // An area with nothing in it gets no tab rather than an empty one: Studio
  // has no protein-structure tools yet, and a tab that is always empty reads
  // as something broken.
  const counts = (value: string) =>
    value === "all" ? (tools?.length ?? 0) : (tools?.filter((t) => t.area === value).length ?? 0);
  const tabs = AREAS.filter(({ value }) => value === "all" || counts(value) > 0);

  const query = search.trim().toLowerCase();
  const hasFilters = query !== "" || decisionsOnly;
  const visible = tools?.filter(
    (tool) =>
      (area === "all" || tool.area === area) &&
      (!query || `${tool.name} ${tool.id} ${tool.description}`.toLowerCase().includes(query)) &&
      (!decisionsOnly || tool.judgment),
  );

  function clearFilters() {
    setSearch("");
    setDecisionsOnly(false);
  }

  return (
    <div className="w-full min-w-0 space-y-6">
      <PageHeader
        title="Tools"
        description="The analysis steps Studio can run, what each one needs, and what it gives back."
      />

      <div className="rounded-sm border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
        A tool takes something and returns something, and the labels on each card say which, so you
        can see which steps can follow which. Tools marked{" "}
        <span className="font-medium text-foreground">Needs a chemist</span> return ranked options
        rather than an answer. That is on purpose: on a benzamide series the highest-coverage core
        reaches it only by dropping the amide carbonyl so a thioamide fits, and the right call is
        the narrower core. Nothing picks the first option for you.
      </div>

      {isLoading && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" aria-label="Loading tools">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}

      {isError && (
        <QueryError title="Could not load tools" retry={() => refetch()} retrying={isFetching} />
      )}

      {tools && (
        <Tabs value={area} onValueChange={setArea} className="space-y-3">
          <TabsList aria-label="Kind of work">
            {tabs.map(({ value, label }) => (
              <TabsTrigger key={value} value={value}>
                {label}
                <span className="rounded bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground">
                  {counts(value)}
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
                aria-label="Search tools"
                placeholder="Search tools…"
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                className="pl-9"
              />
            </div>
            <Button
              variant={decisionsOnly ? "default" : "outline"}
              size="sm"
              aria-pressed={decisionsOnly}
              onClick={() => setDecisionsOnly((on) => !on)}
            >
              Needs a chemist
            </Button>
            {hasFilters && (
              <Button variant="ghost" size="sm" onClick={clearFilters}>
                Clear filters
              </Button>
            )}
          </div>

          <TabsContent value={area} className="space-y-3">
            <output className="block text-xs text-muted-foreground">
              {visible?.length} {visible?.length === 1 ? "tool" : "tools"}
              {hasFilters ? " matching your filters" : " available"}
            </output>

            {visible?.length ? (
              <div className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3">
                {visible.map((tool) => (
                  <ToolCard key={tool.id} tool={tool} />
                ))}
              </div>
            ) : (
              <div className="rounded-sm border border-dashed px-6 py-16 text-center">
                <h2 className="text-sm font-medium">
                  {hasFilters ? "No tools match your filters" : "No tools available"}
                </h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  {hasFilters
                    ? "Try another search or clear your filters to see every tool."
                    : "Tools will appear here when they are added to the catalogue."}
                </p>
                {hasFilters && (
                  <Button variant="outline" className="mt-4" onClick={clearFilters}>
                    Clear filters
                  </Button>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
