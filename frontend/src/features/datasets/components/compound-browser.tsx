"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Badge } from "@/shared/components/ui/badge";
import { Button } from "@/shared/components/ui/button";
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
import { useState } from "react";
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
  const [split, setSplit] = useState<(typeof PARTITIONS)[number] | undefined>();

  const { data, isLoading, isError } = useDatasetCompounds(dataset.id, {
    offset,
    limit: PAGE_SIZE,
    sort: "target",
    sort_dir: descending ? "desc" : "asc",
    split,
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
        <Button variant="ghost" size="sm" onClick={() => reset(() => setDescending((d) => !d))}>
          {dataset.target.column}
          {descending ? <ArrowDown className="size-3.5" /> : <ArrowUp className="size-3.5" />}
        </Button>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[120px]">Structure</TableHead>
            <TableHead>SMILES</TableHead>
            <TableHead className="text-right">{dataset.target.column}</TableHead>
            <TableHead className="w-[110px]">Partition</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading && !data
            ? Array.from({ length: 6 }, (_, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton, no identity
                <TableRow key={index}>
                  <TableCell colSpan={4}>
                    <Skeleton className="h-16 w-full" />
                  </TableCell>
                </TableRow>
              ))
            : data?.items.map((compound) => (
                <TableRow key={compound.structure}>
                  <TableCell>
                    <StructureThumbnail smiles={compound.structure} size={96} />
                  </TableCell>
                  <TableCell className="max-w-[1px] truncate font-mono text-xs text-muted-foreground">
                    {compound.structure}
                  </TableCell>
                  <TableCell className="text-right">
                    <ReadoutValue
                      value={compound.target}
                      unit={dataset.target.unit}
                      precision={3}
                    />
                  </TableCell>
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
