"use client";

import { useFolders } from "@/features/folders";
import { useProtocols } from "@/features/protocols";
import { SegmentedToggle } from "@/shared/components/segmented-toggle";
import { Badge } from "@/shared/components/ui/badge";
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
import type { PredictionCountsWire } from "@/shared/lib/api/model";
import { useMemberName } from "@/shared/lib/auth/use-workspace-members";
import { useUrlParams } from "@/shared/lib/use-url-params";
import { Plus } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { type RunFilters, useRuns } from "../hooks/use-runs";
import { groupRunsByDay } from "../lib/group-runs";
import { RUN_STATUS_COPY, type Run } from "../types";

const ALL = "all";
const STATUS_OPTIONS = [
  { value: "pending,running", label: "In progress" },
  { value: "ready", label: RUN_STATUS_COPY.ready },
  { value: "failed", label: RUN_STATUS_COPY.failed },
  { value: "cancelled", label: RUN_STATUS_COPY.cancelled },
];

function RunRows({
  filters,
  filtered,
  onlyMine,
  onClear,
  onShowAll,
}: {
  filters: RunFilters;
  filtered: boolean;
  onlyMine: boolean;
  onClear: () => void;
  onShowAll: () => void;
}) {
  const [cursor, setCursor] = useState<string | undefined>();
  const [pages, setPages] = useState<Run[]>([]);
  const { data, isLoading, isError } = useRuns("prediction", cursor, filters);
  // ponytail: the newest 200 protocols only; a workspace past that loses older names here
  // (and in the Protocol filter). Upgrade: a published-only filter or a lookup by id.
  const protocols = useProtocols(undefined, 200);
  const memberName = useMemberName();

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  // Resolve the protocol's name so a run is never identified by an id.
  const protocolName = (protocolId: string | null | undefined) =>
    protocols.data?.items.find((protocol) => protocol.id === protocolId)?.name;

  return (
    <>
      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Could not load runs</p>
        </div>
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          {onlyMine ? (
            <>
              <p className="text-sm font-medium">You have no runs yet.</p>
              <Button variant="outline" className="mt-4" onClick={onShowAll}>
                Show all runs
              </Button>
            </>
          ) : filtered ? (
            <>
              <p className="text-sm font-medium">No runs match these filters.</p>
              <Button variant="outline" className="mt-4" onClick={onClear}>
                Clear filters
              </Button>
            </>
          ) : (
            <>
              <p className="text-sm font-medium">No runs yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Publish a protocol, then run it across your own compounds.
              </p>
            </>
          )}
        </div>
      )}

      {items.length > 0 && (
        <div className="space-y-5">
          {groupRunsByDay(items).map((group) => (
            <section key={group.day} className="space-y-1.5">
              <h2 className="text-xs font-medium text-muted-foreground">{group.day}</h2>
              <ul className="divide-y divide-border rounded-lg border border-border">
                {group.runs.map((run) => {
                  const scored = (run.metrics as Partial<PredictionCountsWire> | null)?.scored_rows;
                  const protocol = protocolName(run.protocol_id);
                  const name = run.name ?? protocol ?? "Prediction run";
                  return (
                    <li key={run.id}>
                      <Link
                        href={`/runs/${run.id}`}
                        className="flex items-center gap-3 px-3 py-2 text-sm transition-colors hover:bg-muted/40"
                      >
                        <span className="shrink-0 tabular-nums text-muted-foreground">
                          {new Date(run.created_at).toLocaleTimeString("en-US", {
                            hour: "numeric",
                            minute: "2-digit",
                          })}
                        </span>
                        <span className="min-w-0 flex-1 truncate">
                          <span className="font-medium">{name}</span>
                          {protocol && name !== protocol && (
                            <span className="hidden text-muted-foreground sm:inline">
                              {" "}
                              · {protocol}
                            </span>
                          )}
                          {scored != null && (
                            <span className="text-muted-foreground">
                              {" "}
                              · {scored.toLocaleString("en-US")} compound
                              {scored === 1 ? "" : "s"}
                            </span>
                          )}
                          {run.source && (
                            <span className="text-muted-foreground"> · ChemCellar</span>
                          )}
                        </span>
                        <span className="hidden max-w-32 shrink-0 truncate text-xs text-muted-foreground sm:inline">
                          {memberName(run.requested_by)}
                        </span>
                        <Badge
                          variant={run.status === "ready" ? "default" : "outline"}
                          className="shrink-0 font-normal"
                        >
                          {RUN_STATUS_COPY[run.status] ?? run.status}
                        </Badge>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      )}

      {data?.next_cursor && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            onClick={() => {
              setPages(items);
              setCursor(data.next_cursor ?? undefined);
            }}
          >
            Load more
          </Button>
        </div>
      )}
    </>
  );
}

export function RunList() {
  const { params, set } = useUrlParams();
  // ponytail: newest 200 protocols only, as in RunRows. Upgrade: a published-only filter.
  const protocols = useProtocols(undefined, 200);
  const folders = useFolders("protocol");

  // Mine is the default: only an explicit `mine=0` shows everyone's runs.
  const mine = params.get("mine") !== "0";
  const protocolId = params.get("protocol") ?? undefined;
  const status = params.get("status") ?? undefined;
  const folderId = params.get("folder") ?? undefined;
  const q = params.get("q") ?? "";

  // The box is typed into freely; the URL (and so the request) follows 300 ms later.
  const [search, setSearch] = useState(q);
  const setRef = useRef(set);
  setRef.current = set;
  useEffect(() => {
    if (search === q) return;
    const timer = setTimeout(() => setRef.current({ q: search.trim() || undefined }), 300);
    return () => clearTimeout(timer);
  }, [search, q]);

  const filtered = !mine || Boolean(protocolId || status || folderId || q);
  const clear = () => {
    setSearch("");
    set({
      mine: undefined,
      protocol: undefined,
      status: undefined,
      folder: undefined,
      q: undefined,
    });
  };

  const filters: RunFilters = {
    mine,
    protocolId,
    folderId,
    q: q || undefined,
    statuses: status?.split(","),
  };
  const published = (protocols.data?.items ?? []).filter((protocol) => protocol.status !== "draft");

  return (
    <div className="mx-auto w-full max-w-4xl space-y-4 p-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold">Runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Prediction runs apply a published protocol to a set of compounds. Training runs are
            listed on their protocol.
          </p>
        </div>
        <Button asChild>
          <Link href="/runs/new">
            <Plus className="size-4" />
            Run a protocol
          </Link>
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <SegmentedToggle
          label="Run owner"
          options={[
            { value: "all", label: "All" },
            { value: "mine", label: "Mine" },
          ]}
          value={mine ? "mine" : "all"}
          onChange={(value) => set({ mine: value === "mine" ? undefined : "0" })}
        />

        <Select
          value={protocolId ?? ALL}
          onValueChange={(value) => set({ protocol: value === ALL ? undefined : value })}
        >
          <SelectTrigger size="sm" aria-label="Protocol" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All protocols</SelectItem>
            {published.map((protocol) => (
              <SelectItem key={protocol.id} value={protocol.id}>
                {protocol.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={status ?? ALL}
          onValueChange={(value) => set({ status: value === ALL ? undefined : value })}
        >
          <SelectTrigger size="sm" aria-label="Status" className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All statuses</SelectItem>
            {STATUS_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={folderId ?? ALL}
          onValueChange={(value) => set({ folder: value === ALL ? undefined : value })}
        >
          <SelectTrigger size="sm" aria-label="Folder" className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All folders</SelectItem>
            {folders.data?.items.map((folder) => (
              <SelectItem key={folder.id} value={folder.id}>
                {folder.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          type="search"
          aria-label="Search runs or protocols"
          placeholder="Search runs or protocols"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-8 w-48"
        />

        {filtered && (
          <Button variant="link" size="sm" onClick={clear}>
            Clear filters
          </Button>
        )}
      </div>

      <RunRows
        key={JSON.stringify(filters)}
        filters={filters}
        filtered={filtered}
        onlyMine={mine && !protocolId && !status && !folderId && !q}
        onClear={clear}
        onShowAll={() => set({ mine: "0" })}
      />
    </div>
  );
}
