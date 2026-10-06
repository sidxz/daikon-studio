"use client";

import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import type { FolderResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { ChevronDown, Plus } from "lucide-react";
import { type DragEvent, type ReactNode, useLayoutEffect, useRef, useState } from "react";

const tab = "flex min-h-10 items-center gap-2 border-b-2 px-3 py-2 text-sm font-medium";

/** Fits complete folder names into the available space, keeping the selected one visible. */
export function FolderTabsBar({
  all,
  folders,
  activeId,
  canEdit,
  onSelect,
  onCreate,
  renderFolder,
  onMoreDragOver,
}: {
  all: string;
  folders: FolderResponse[];
  activeId: string | undefined;
  canEdit: boolean;
  onSelect: (id: string | undefined) => void;
  onCreate: () => void;
  renderFolder: (folder: FolderResponse, inMenu?: boolean, afterSelect?: () => void) => ReactNode;
  onMoreDragOver: (event: DragEvent) => boolean;
}) {
  const container = useRef<HTMLDivElement>(null);
  const measurements = useRef<HTMLDivElement>(null);
  const [visibleIds, setVisibleIds] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  useLayoutEffect(() => {
    const root = container.current;
    const probes = measurements.current;
    if (!root || !probes) return;
    const fit = () => {
      const widths = Array.from(probes.children).map(
        (element) => element.getBoundingClientRect().width,
      );
      const budget = root.clientWidth - widths[0] - widths[1] - 12;
      const folderWidths = widths.slice(3);
      const total = folderWidths.reduce((sum, width) => sum + width + 4, 0);
      const available = total <= budget ? budget : budget - widths[2] - 4;
      const selectedIndex = folders.findIndex((folder) => folder.id === activeId);
      const indices: number[] = selectedIndex >= 0 ? [selectedIndex] : [];
      let used = selectedIndex >= 0 ? folderWidths[selectedIndex] + 4 : 0;
      for (let index = 0; index < folders.length; index++) {
        if (index === selectedIndex) continue;
        const width = folderWidths[index] + 4;
        if (used + width > available && indices.length > 0) break;
        indices.push(index);
        used += width;
      }
      const ids = indices.sort((a, b) => a - b).map((index) => folders[index].id);
      setVisibleIds((previous) => (previous.join("\0") === ids.join("\0") ? previous : ids));
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(root);
    observer.observe(probes);
    return () => observer.disconnect();
  }, [folders, activeId]);

  const visible = folders.filter((folder) => visibleIds.includes(folder.id));
  const overflow = folders.filter((folder) => !visibleIds.includes(folder.id));
  const matches = overflow.filter((folder) =>
    folder.name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div
      ref={container}
      className="relative flex min-w-0 flex-wrap items-center gap-x-2 border-b border-border"
    >
      <ul className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
        <li className="basis-full shrink-0 sm:basis-auto">
          <button
            type="button"
            aria-pressed={!activeId}
            onClick={() => onSelect(undefined)}
            className={cn(
              tab,
              "whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              !activeId
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {all}
          </button>
        </li>
        {visible.map((folder) => renderFolder(folder))}
        {overflow.length > 0 && (
          <li className="shrink-0">
            <Popover
              open={open}
              onOpenChange={(next) => {
                setOpen(next);
                if (!next) setSearch("");
              }}
            >
              <PopoverTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  aria-label="More folders"
                  onDragOver={(event) => {
                    if (onMoreDragOver(event)) setOpen(true);
                  }}
                >
                  More <ChevronDown className="size-3.5" />
                </Button>
              </PopoverTrigger>
              <PopoverContent align="start" className="w-80 max-w-[calc(100vw-2rem)] p-2">
                <Input
                  aria-label="Search folders"
                  placeholder="Search folders…"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
                <ul className="mt-2 max-h-72 overflow-y-auto">
                  {matches.map((folder) =>
                    renderFolder(folder, true, () => {
                      setOpen(false);
                      setSearch("");
                    }),
                  )}
                </ul>
                {matches.length === 0 && (
                  <p className="px-3 py-4 text-sm text-muted-foreground">No folders match.</p>
                )}
              </PopoverContent>
            </Popover>
          </li>
        )}
      </ul>
      {canEdit && (
        <Button
          variant="ghost"
          size="icon"
          aria-label="New folder"
          className="mt-1 ml-auto h-8 w-8 shrink-0 self-start text-muted-foreground sm:w-auto sm:px-2.5"
          onClick={onCreate}
        >
          <Plus className="size-3.5" /> <span className="hidden sm:inline">New folder</span>
        </Button>
      )}
      {/* Non-interactive width probes stay clipped and out of the accessible tree. */}
      <div aria-hidden className="pointer-events-none absolute h-0 w-0 overflow-hidden">
        <div ref={measurements} className="flex w-max">
          <span className={cn(tab, "whitespace-nowrap")}>{all}</span>
          <span
            className={
              canEdit
                ? "flex w-8 items-center justify-center gap-2 text-sm font-semibold sm:w-auto sm:px-2.5"
                : "w-0"
            }
          >
            {canEdit && (
              <>
                <Plus className="size-3.5" />
                <span className="hidden sm:inline">New folder</span>
              </>
            )}
          </span>
          <span className="flex items-center gap-1.5 px-2.5 text-sm font-semibold">
            More
            <ChevronDown className="size-3.5" />
          </span>
          {folders.map((folder) => (
            <span
              key={folder.id}
              className="flex shrink-0 items-center gap-2 whitespace-nowrap px-3 py-2 text-sm font-medium"
            >
              {folder.name}
              <span className="text-xs tabular-nums">{folder.item_count}</span>
              {canEdit && <span className="w-7" />}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
