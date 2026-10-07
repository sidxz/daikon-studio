"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { ArrowDown, ArrowUp } from "lucide-react";
import { useEffect, useState } from "react";
import { useDatasetCompounds } from "../hooks/use-datasets";
import type { Dataset } from "../types";

const PAGE_SIZE = 25;
const PARTITIONS = ["train", "validation", "test"] as const;

/**
 * The frozen snapshot's own rows.
 *
 * Until this existed a scientist could see that 1,128 compounds survived
 * validation and could not look at one of them. Sorting is limited to the target
 * and the partition because those are the two orders that mean anything --
 * sorting by SMILES is alphabetical nonsense dressed up as chemistry, and the
 * backend refuses it for the same reason.
 */
export function CompoundBrowser({ dataset }: { dataset: Dataset }) {
  const [offset, setOffset] = useState(0);
  const [descending, setDescending] = useState(false);
  const [sortTarget, setSortTarget] = useState(0);
  const [split, setSplit] = useState<(typeof PARTITIONS)[number] | undefined>();
  // What the search box shows, and what was last sent: one request per pause
  // in typing, not per keystroke, since each reads the whole snapshot.
  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQ(search.trim());
      setOffset(0);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  const { data, isLoading, isError } = useDatasetCompounds(dataset.id, {
    offset,
    limit: PAGE_SIZE,
    sort: "target",
    target: sortTarget,
    sort_dir: descending ? "desc" : "asc",
    split,
    q: q || undefined,
  });

  function reset(next: () => void) {
    setOffset(0);
    next();
  }

  if (isError) {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        Could not read this dataset&apos;s snapshot.
      </div>
    );
  }

  const total = data?.total ?? 0;
  const showing = data?.items.length ?? 0;

  return (
    <div className="space-y-3">
      {dataset.id_column && (
        <Input
          type="search"
          aria-label="Search by ID"
          placeholder={`Search by ${dataset.id_column}`}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="h-8 max-w-xs"
        />
      )}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1">
          <Button
            variant={split === undefined ? "secondary" : "ghost"}
            size="sm"
            onClick={() => reset(() => setSplit(undefined))}
          >
            All
          </Button>
          {PARTITIONS.map((partition) => (
            <Button
              key={partition}
              variant={split === partition ? "secondary" : "ghost"}
              size="sm"
              onClick={() => reset(() => setSplit(partition))}
            >
              {partition}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1">
          {dataset.targets.map((target, index) => (
            <Button
              key={target.column}
              variant={index === sortTarget ? "secondary" : "ghost"}
              size="sm"
              onClick={() =>
                reset(() => {
                  if (index === sortTarget) setDescending((d) => !d);
                  else {
                    setSortTarget(index);
                    setDescending(false);
                  }
                })
              }
            >
              {target.column}
              {index === sortTarget &&
                (descending ? (
                  <ArrowDown className="size-3.5" />
                ) : (
                  <ArrowUp className="size-3.5" />
                ))}
            </Button>
          ))}
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[120px]">Structure</TableHead>
            {dataset.id_column && <TableHead>{dataset.id_column}</TableHead>}
            <TableHead>SMILES</TableHead>
            {dataset.targets.map((target) => (
              <TableHead key={target.column} className="text-right">
                {target.column}
              </TableHead>
            ))}
            <TableHead className="w-[110px]">Partition</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading && !data
            ? Array.from({ length: 6 }, (_, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton, no identity
                <TableRow key={index}>
                  <TableCell colSpan={3 + dataset.targets.length + (dataset.id_column ? 1 : 0)}>
                    <Skeleton className="h-16 w-full" />
                  </TableCell>
                </TableRow>
              ))
            : data?.items.map((compound) => (
                <TableRow key={compound.structure}>
                  <TableCell>
                    <StructureThumbnail smiles={compound.structure} size={96} />
                  </TableCell>
                  {dataset.id_column && (
                    <TableCell className="font-mono text-xs">
                      {compound.compound_id ?? "N/A"}
                    </TableCell>
                  )}
                  <TableCell className="max-w-[1px] truncate font-mono text-xs text-muted-foreground">
                    {compound.structure}
                  </TableCell>
                  {dataset.targets.map((target) => (
                    <TableCell key={target.column} className="text-right">
                      <ReadoutValue
                        value={compound.targets[target.column] ?? null}
                        unit={target.unit}
                        precision={3}
                      />
                    </TableCell>
                  ))}
                  <TableCell>
                    <Badge variant="outline" className="font-normal">
                      {compound.split}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
        </TableBody>
      </Table>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {total === 0
            ? "No compounds"
            : `${(offset + 1).toLocaleString()}–${(offset + showing).toLocaleString()} of ${total.toLocaleString()}`}
        </span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={offset === 0}
            onClick={() => setOffset((current) => Math.max(0, current - PAGE_SIZE))}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={offset + showing >= total}
            onClick={() => setOffset((current) => current + PAGE_SIZE)}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
