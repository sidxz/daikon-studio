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
import { useDatasetProfile } from "../hooks/use-datasets";
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
  // A sequence dataset has no 2D depiction to draw and no SMILES to name. Drawing
  // one anyway hands RDKit a protein and renders whatever comes back.
  const isSequence = dataset.validation_report?.structure_kind === "sequence";
  // The parent sequence, so a row can show what it changed instead of 286 characters
  // truncated at the same place for every variant. Shares react-query's cache with the
  // Diversity tab, so this costs no extra request.
  const profile = useDatasetProfile(isSequence ? dataset.id : undefined);
  const consensus =
    profile.data && "variants" in profile.data ? profile.data.variants?.consensus : undefined;
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
            {!isSequence && <TableHead className="w-[120px]">Structure</TableHead>}
            {dataset.id_column && <TableHead>{dataset.id_column}</TableHead>}
            <TableHead>{isSequence ? "Sequence" : "SMILES"}</TableHead>
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
                  <TableCell
                    colSpan={
                      // structure thumbnail (molecules only) + text + partition
                      (isSequence ? 2 : 3) + dataset.targets.length + (dataset.id_column ? 1 : 0)
                    }
                  >
                    <Skeleton className="h-16 w-full" />
                  </TableCell>
                </TableRow>
              ))
            : data?.items.map((compound) => (
                <TableRow key={compound.structure}>
                  {!isSequence && (
                    <TableCell>
                      <StructureThumbnail smiles={compound.structure} size={96} />
                    </TableCell>
                  )}
                  {dataset.id_column && (
                    <TableCell className="font-mono text-xs">
                      {compound.compound_id ?? "N/A"}
                    </TableCell>
                  )}
                  <TableCell className="max-w-[1px] truncate font-mono text-xs text-muted-foreground">
                    {consensus ? (
                      <MutationInContext sequence={compound.structure} consensus={consensus} />
                    ) : (
                      compound.structure
                    )}
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

/**
 * A variant shown as what it changed, not as its first forty residues.
 *
 * Every variant of one parent is identical for hundreds of characters, so a truncated
 * sequence column renders every row the same -- technically the data, and useless. This
 * diffs against the parent and shows the substitution in context. Falling back to the
 * raw text matters: a row differing at many positions has no single mutation to point
 * at, and inventing one would be worse than showing the sequence.
 */
function MutationInContext({ sequence, consensus }: { sequence: string; consensus: string }) {
  if (sequence.length !== consensus.length) return <>{sequence}</>;
  const differing: number[] = [];
  for (let i = 0; i < sequence.length && differing.length < 3; i++) {
    if (sequence[i] !== consensus[i]) differing.push(i);
  }
  if (differing.length === 0) return <span className="text-muted-foreground">parent sequence</span>;
  if (differing.length > 2) return <>{sequence}</>;

  const flank = 6;
  return (
    <span className="whitespace-nowrap">
      {differing.map((index, order) => (
        <span key={index}>
          {order > 0 && <span className="px-1 text-muted-foreground">·</span>}
          <span className="text-muted-foreground">
            {index > flank ? "…" : ""}
            {consensus.slice(Math.max(0, index - flank), index)}
          </span>
          <span className="rounded bg-primary/10 px-1 font-semibold text-foreground">
            {consensus[index]}
            {index + 1}
            {sequence[index]}
          </span>
          <span className="text-muted-foreground">
            {consensus.slice(index + 1, index + 1 + flank)}
            {index + 1 + flank < consensus.length ? "…" : ""}
          </span>
        </span>
      ))}
    </span>
  );
}
