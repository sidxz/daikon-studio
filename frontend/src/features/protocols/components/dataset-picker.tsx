"use client";

import type { Dataset } from "@/features/datasets";
import { Button } from "@/shared/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/shared/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import { Check, ChevronsUpDown, Loader2 } from "lucide-react";
import { useState } from "react";

export function DatasetPicker({
  items,
  selected,
  search,
  onSearch,
  onSelect,
  loading,
  failed,
}: {
  items: Dataset[];
  selected: Dataset | undefined;
  search: string;
  onSearch: (value: string) => void;
  onSelect: (value: string) => void;
  loading: boolean;
  failed: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id="training-dataset"
          variant="outline"
          aria-label="Dataset"
          aria-expanded={open}
          aria-controls="training-dataset-options"
          className="h-auto min-h-10 w-full justify-between text-left font-normal"
        >
          <span className="truncate">{selected?.name ?? "Choose a dataset"}</span>
          <ChevronsUpDown className="ml-2 size-4 shrink-0 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0" align="start">
        <Command shouldFilter={false}>
          <CommandInput
            value={search}
            onValueChange={onSearch}
            placeholder="Search all datasets…"
            aria-label="Search datasets"
          />
          <CommandList id="training-dataset-options">
            {loading ? (
              <div
                aria-live="polite"
                className="flex items-center gap-2 p-4 text-sm text-muted-foreground"
              >
                <Loader2 className="size-4 animate-spin" />
                Loading datasets…
              </div>
            ) : failed ? (
              <div aria-live="polite" className="p-4 text-sm text-destructive">
                Could not load datasets. Use Try again on the page.
              </div>
            ) : (
              <>
                <CommandEmpty>No matching datasets. Try another name.</CommandEmpty>
                <CommandGroup>
                  {items.map((item) => (
                    <CommandItem
                      key={item.id}
                      value={item.id}
                      onSelect={() => {
                        onSelect(item.id);
                        setOpen(false);
                      }}
                    >
                      <Check
                        className={`size-4 shrink-0 ${selected?.id === item.id ? "opacity-100" : "opacity-0"}`}
                      />
                      <div className="min-w-0">
                        <p className="truncate text-sm">{item.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {item.row_count.toLocaleString()} compounds · {item.targets?.length ?? 0}{" "}
                          targets · {item.split?.strategy ?? ""} split
                        </p>
                      </div>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
