"use client";

import { useFolders } from "@/features/folders";
import { useProtocolOptions } from "@/features/protocols";
import { RunsIcon } from "@/shared/components/icons/nav-icons";
import { InitialsAvatar } from "@/shared/components/initials-avatar";
import { LogoMark } from "@/shared/components/logo-mark";
import { PageHeader } from "@/shared/components/page-header";
import { QueryError } from "@/shared/components/query-error";
import { SegmentedToggle } from "@/shared/components/segmented-toggle";
import { StatusDot, type StatusTone } from "@/shared/components/status-dot";
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
import { cn } from "@/shared/lib/utils";
import { ChevronRight, Search, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useRunOwner } from "../hooks/use-run-owner";
import { type RunFilters, useRuns } from "../hooks/use-runs";
import { dayLabel, groupRunsByDay } from "../lib/group-runs";
import { runDateBounds } from "../lib/run-date-range";
import { RUN_STATUS_COPY, type Run } from "../types";
import { ProtocolPicker } from "./protocol-picker";
import { RunDateFilter } from "./run-date-filter";

const ALL = "all";
const STATUS_OPTIONS = [
  { value: "pending,running", label: "In progress" },
  { value: "ready", label: RUN_STATUS_COPY.ready },
  { value: "failed", label: RUN_STATUS_COPY.failed },
  { value: "cancelled", label: RUN_STATUS_COPY.cancelled },
];

/** The dot's tone and the word's color. Only work in progress is drawn in the foreground. */
const STATUS_LOOK: Record<string, { tone: StatusTone; word: string }> = {
  pending: { tone: "active", word: "text-foreground" },
  running: { tone: "active", word: "text-foreground" },
  ready: { tone: "success", word: "text-muted-foreground" },
  failed: { tone: "failed", word: "text-destructive" },
  cancelled: { tone: "muted", word: "text-muted-foreground" },
};

function RunStatus({
  status,
  phase,
  progress,
}: { status: string; phase?: string | null; progress?: number }) {
  const look = STATUS_LOOK[status] ?? STATUS_LOOK.cancelled;
  return (
    <span className="flex min-w-0 flex-col justify-self-end [grid-area:status] sm:justify-self-start">
      <span className="flex items-center gap-2">
        <StatusDot tone={look.tone} />
        <span className={look.word}>{RUN_STATUS_COPY[status] ?? status}</span>
      </span>
      {status === "running" && (
        <span
          className="mt-0.5 max-w-36 truncate text-xs text-muted-foreground"
          title={phase ?? undefined}
        >
          {phase ?? "Predicting"}
          {progress != null && ` · ${Math.round(progress * 100)}%`}
        </span>
      )}
      {status === "pending" && (
        <span className="mt-0.5 text-xs text-muted-foreground">Waiting for a runner</span>
      )}
    </span>
  );
}

// The header and rows share columns. Protocol and source stay with the run name.
const COLUMNS =
  "sm:grid-cols-[4.75rem_minmax(0,1fr)_5.5rem_2rem_7.5rem_1rem] lg:grid-cols-[4.75rem_minmax(0,1fr)_6rem_10rem_8rem_1rem]";
// Below sm: run and status, then time, compounds and the navigation affordance.
const ROW = cn(
  "grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-x-3 gap-y-1 [grid-template-areas:'name_name_status''time_count_action']",
  COLUMNS,
  "sm:[grid-template-areas:'time_name_count_by_status_action']",
);

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
  const { data, isLoading, isError, refetch, isFetching } = useRuns("prediction", cursor, filters);
  const protocols = useProtocolOptions();
  const memberName = useMemberName();

  const items = cursor ? [...pages, ...(data?.items ?? [])] : (data?.items ?? []);

  // Resolve the protocol's name so a run is never identified by an id.
  const protocolName = (protocolId: string | null | undefined) =>
    protocols.data?.find((protocol) => protocol.id === protocolId)?.name;

  const now = new Date();

  return (
    <>
      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
        </div>
      )}

      {isError && (
        <QueryError title="Could not load runs" retry={() => refetch()} retrying={isFetching} />
      )}

      {data && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-10 text-center">
          <RunsIcon className="mx-auto mb-3 size-6 text-icon-runs opacity-60" />
          {onlyMine ? (
            <>
              <p className="text-sm font-medium">You have no runs yet.</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Apply a published protocol to your compounds to get started.
              </p>
              <Button asChild className="mt-4 mr-2">
                <Link href="/runs/new">Run a protocol</Link>
              </Button>
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
              <Button asChild className="mt-4">
                <Link href="/runs/new">Run a protocol</Link>
              </Button>
            </>
          )}
        </div>
      )}

      {items.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div
            aria-hidden
            className={cn(
              "hidden items-center gap-x-3 border-b border-border bg-muted/35 px-4 py-2.5 text-xs font-medium text-muted-foreground sm:grid",
              COLUMNS,
            )}
          >
            <span>Time</span>
            <span>Run / protocol</span>
            <span className="text-right">Compounds</span>
            <span>Started by</span>
            <span>Status</span>
            <span />
          </div>
          <div>
            {groupRunsByDay(items, (iso) => dayLabel(iso, now)).map((group) => (
              <section key={group.day} className="border-t border-border first:border-t-0">
                <div className="flex items-center gap-2 border-b border-border/60 bg-muted/15 px-4 py-2">
                  <h2 className="text-xs font-semibold">{group.day}</h2>
                  <span className="text-xs text-muted-foreground" title="Runs loaded for this day">
                    {group.runs.length} shown
                  </span>
                </div>
                <ul className="divide-y divide-border/60">
                  {group.runs.map((run) => {
                    const scored = (run.metrics as Partial<PredictionCountsWire> | null)
                      ?.scored_rows;
                    const protocol = protocolName(run.protocol_id);
                    const name = run.name ?? protocol ?? "Prediction run";
                    const otherProtocol = protocol !== name ? protocol : undefined;
                    const context = [
                      otherProtocol,
                      run.source && `ChemCellar · ${run.source.protocol_name}`,
                    ]
                      .filter(Boolean)
                      .join(" · ");
                    const requester = memberName(run.requested_by);
                    return (
                      <li key={run.id}>
                        <Link
                          href={`/runs/${run.id}`}
                          className={cn(
                            ROW,
                            "group min-h-[3.25rem] px-4 py-2.5 text-sm transition-colors hover:bg-accent/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                          )}
                        >
                          <span className="tabular-nums text-muted-foreground [grid-area:time]">
                            {new Date(run.created_at).toLocaleTimeString("en-US", {
                              hour: "numeric",
                              minute: "2-digit",
                            })}
                          </span>
                          <span className="min-w-0 [grid-area:name]">
                            <span className="block truncate font-medium" title={name}>
                              {name}
                            </span>
                            {context && (
                              <span
                                className="mt-0.5 block truncate text-xs text-muted-foreground"
                                title={context}
                              >
                                {context}
                              </span>
                            )}
                          </span>
                          <span className="text-right tabular-nums [grid-area:count]">
                            {scored != null ? (
                              <>
                                {scored.toLocaleString("en-US")}
                                <span className="sm:sr-only">
                                  {" "}
                                  compound{scored === 1 ? "" : "s"}
                                </span>
                              </>
                            ) : (
                              <span
                                className="text-muted-foreground"
                                aria-label="Compound count unavailable"
                              >
                                —
                              </span>
                            )}
                          </span>
                          <span className="hidden min-w-0 items-center gap-2 [grid-area:by] sm:flex">
                            <InitialsAvatar name={requester} id={run.requested_by} />
                            {/* The avatar already names them to a screen reader. */}
                            <span
                              aria-hidden
                              className="hidden max-w-32 truncate text-muted-foreground lg:inline"
                            >
                              {requester}
                            </span>
                          </span>
                          <RunStatus
                            status={run.status}
                            phase={run.phase}
                            progress={run.progress}
                          />
                          <ChevronRight
                            aria-hidden
                            className="size-3.5 justify-self-end text-muted-foreground/50 [grid-area:action] sm:opacity-0 sm:transition-opacity sm:group-hover:opacity-100 sm:group-focus-visible:opacity-100"
                          />
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
          <div className="border-t border-border bg-muted/15 px-4 py-2 text-xs text-muted-foreground">
            {items.length.toLocaleString()} run{items.length === 1 ? "" : "s"} shown
          </div>
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
  const protocols = useProtocolOptions();
  const folders = useFolders("protocol");

  const { mine, ready: ownerReady, rememberOwner } = useRunOwner(params.get("mine"));
  const protocolId = params.get("protocol") ?? undefined;
  const status = params.get("status") ?? undefined;
  const folderId = params.get("folder") ?? undefined;
  const q = params.get("q") ?? "";
  const from = params.get("from") ?? "";
  const to = params.get("to") ?? "";

  // The box is typed into freely; the URL (and so the request) follows 300 ms later.
  const [search, setSearch] = useState(q);
  const setRef = useRef(set);
  setRef.current = set;
  useEffect(() => {
    if (search === q) return;
    const timer = setTimeout(() => setRef.current({ q: search.trim() || undefined }), 300);
    return () => clearTimeout(timer);
  }, [search, q]);

  const filtered = Boolean(protocolId || status || folderId || q || from || to);
  const chooseOwner = (mine: boolean) => {
    rememberOwner(mine);
    set({ mine: mine ? "1" : "0" });
  };
  const clear = () => {
    setSearch("");
    set({
      protocol: undefined,
      status: undefined,
      folder: undefined,
      q: undefined,
      from: undefined,
      to: undefined,
    });
  };

  const filters: RunFilters = {
    mine,
    protocolId,
    folderId,
    q: q || undefined,
    statuses: status?.split(","),
    ...runDateBounds(from, to),
  };
  const published = (protocols.data ?? []).filter((protocol) => protocol.status !== "draft");

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader
        title="Runs"
        description="Review prediction results. Training history is on each protocol."
        action={
          <Button asChild>
            <Link href="/runs/new">
              <LogoMark className="-mx-1 size-7" />
              Run a protocol
            </Link>
          </Button>
        }
      />

      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card p-3">
        <div className="relative order-first min-w-0 basis-full sm:min-w-48 sm:flex-1 sm:basis-48">
          <Search
            aria-hidden
            className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            type="search"
            aria-label="Search runs or protocols"
            placeholder="Search runs or protocols…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-9 pl-9"
          />
        </div>
        <SegmentedToggle
          label="Run owner"
          options={[
            { value: "all", label: "All" },
            { value: "mine", label: "Started by me" },
          ]}
          value={mine ? "mine" : "all"}
          onChange={(value) => chooseOwner(value === "mine")}
        />

        <ProtocolPicker
          protocols={published}
          value={protocolId ?? ""}
          onChange={(id) => set({ protocol: id || undefined })}
          allLabel="All protocols"
          loading={protocols.isLoading}
          className="h-9 w-full sm:w-48"
        />

        <Select
          value={status ?? ALL}
          onValueChange={(value) => set({ status: value === ALL ? undefined : value })}
        >
          <SelectTrigger aria-label="Status" className="min-w-32 flex-1 sm:w-32 sm:flex-none">
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
          <SelectTrigger aria-label="Folder" className="min-w-32 flex-1 sm:w-36 sm:flex-none">
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

        <RunDateFilter
          from={from}
          to={to}
          onChange={(from, to) => set({ from: from || undefined, to: to || undefined })}
        />

        {filtered && (
          <Button variant="ghost" size="sm" onClick={clear}>
            <X className="size-4" />
            Clear
          </Button>
        )}
      </div>

      {ownerReady ? (
        <RunRows
          key={JSON.stringify(filters)}
          filters={filters}
          filtered={filtered}
          onlyMine={mine && !filtered}
          onClear={clear}
          onShowAll={() => chooseOwner(false)}
        />
      ) : (
        <Skeleton className="h-40 w-full" />
      )}
    </div>
  );
}
