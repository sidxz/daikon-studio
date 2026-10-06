"use client";

import { useEngines } from "@/features/engines";
import { SegmentedToggle } from "@/shared/components/segmented-toggle";
import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Search, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ProtocolFilters as FilterValues } from "../hooks/use-protocols";

const ALL = "all";

export function ProtocolFilters({
  filters,
  onChange,
}: {
  filters: FilterValues;
  onChange: (values: Record<string, string | undefined>) => void;
}) {
  const q = filters.q ?? "";
  const [search, setSearch] = useState(q);
  const engines = useEngines().data ?? [];
  const change = useRef(onChange);
  change.current = onChange;
  const committed = useRef<string | null>(null);

  useEffect(() => {
    if (committed.current === q) {
      committed.current = null;
      return;
    }
    setSearch(q);
  }, [q]);

  useEffect(() => {
    const text = search.trim();
    if (text === q) return;
    const timer = setTimeout(() => {
      committed.current = text;
      change.current({ q: text || undefined });
    }, 300);
    return () => clearTimeout(timer);
  }, [search, q]);

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card p-3">
      <div className="relative min-w-0 basis-full sm:flex-1 sm:basis-48">
        <Search
          aria-hidden
          className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          type="search"
          aria-label="Search protocols or targets"
          placeholder="Search protocols or targets…"
          maxLength={256}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="h-9 pl-9"
        />
      </div>
      <Select
        value={filters.engineId ?? ALL}
        onValueChange={(value) => onChange({ engine_id: value === ALL ? undefined : value })}
      >
        <SelectTrigger aria-label="Engine" className="min-w-40 flex-1 sm:w-52 sm:flex-none">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All engines</SelectItem>
          {filters.engineId && !engines.some((engine) => engine.id === filters.engineId) && (
            <SelectItem value={filters.engineId}>{filters.engineId}</SelectItem>
          )}
          {engines.map((engine) => (
            <SelectItem key={engine.id} value={engine.id}>
              {engine.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value={filters.status ?? ALL}
        onValueChange={(value) => onChange({ status: value === ALL ? undefined : value })}
      >
        <SelectTrigger
          aria-label="Protocol status"
          className="min-w-32 flex-1 sm:w-36 sm:flex-none"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All statuses</SelectItem>
          <SelectItem value="draft">Draft</SelectItem>
          <SelectItem value="published">Published</SelectItem>
        </SelectContent>
      </Select>
      <SegmentedToggle
        label="Protocol owner"
        options={[
          { value: "all", label: "All" },
          { value: "mine", label: "Created by me" },
        ]}
        value={filters.mine ? "mine" : "all"}
        onChange={(value) => onChange({ mine: value === "mine" ? "1" : undefined })}
      />
      {(search || filters.engineId || filters.status || filters.mine) && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            committed.current = "";
            setSearch("");
            onChange({ q: undefined, engine_id: undefined, status: undefined, mine: undefined });
          }}
        >
          <X className="size-4" />
          Clear
        </Button>
      )}
    </div>
  );
}
