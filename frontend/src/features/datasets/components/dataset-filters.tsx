"use client";

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
import type { SplitStrategy, TargetKind } from "../types";

const ALL = "all";

export function DatasetFilters({
  q,
  targetKind,
  splitStrategy,
  onChange,
}: {
  q: string;
  targetKind: TargetKind | undefined;
  splitStrategy: SplitStrategy | undefined;
  onChange: (values: Record<string, string | undefined>) => void;
}) {
  const [search, setSearch] = useState(q);
  const change = useRef(onChange);
  change.current = onChange;
  const committed = useRef<string | null>(null);

  // External URL changes restore the field; our own commits keep typing uninterrupted.
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
          aria-label="Search datasets or targets"
          placeholder="Search datasets or targets…"
          maxLength={256}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="h-9 pl-9"
        />
      </div>
      <Select
        value={targetKind ?? ALL}
        onValueChange={(value) => onChange({ target_kind: value === ALL ? undefined : value })}
      >
        <SelectTrigger aria-label="Target type" className="min-w-36 flex-1 sm:w-44 sm:flex-none">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All target types</SelectItem>
          <SelectItem value="numeric">Numeric targets</SelectItem>
          <SelectItem value="binary">Binary targets</SelectItem>
        </SelectContent>
      </Select>
      <Select
        value={splitStrategy ?? ALL}
        onValueChange={(value) => onChange({ split_strategy: value === ALL ? undefined : value })}
      >
        <SelectTrigger aria-label="Split strategy" className="min-w-32 flex-1 sm:w-40 sm:flex-none">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All splits</SelectItem>
          <SelectItem value="scaffold">Scaffold split</SelectItem>
          <SelectItem value="random">Random split</SelectItem>
        </SelectContent>
      </Select>
      {(search || targetKind || splitStrategy) && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            committed.current = "";
            setSearch("");
            onChange({ q: undefined, target_kind: undefined, split_strategy: undefined });
          }}
        >
          <X className="size-4" />
          Clear
        </Button>
      )}
    </div>
  );
}
