import { Button } from "@/shared/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/shared/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import type { ProtocolResponse } from "@/shared/lib/api/model";
import { cn } from "@/shared/lib/utils";
import { Check, ChevronsUpDown } from "lucide-react";
import { useState } from "react";

export function ProtocolPicker({
  protocols,
  value,
  onChange,
  loading,
  allLabel,
  className,
  id,
}: {
  protocols: ProtocolResponse[];
  value: string;
  onChange: (id: string) => void;
  loading?: boolean;
  allLabel?: string;
  className?: string;
  id?: string;
}) {
  const [open, setOpen] = useState(false);
  const selected = protocols.find((protocol) => protocol.id === value);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          variant="outline"
          // biome-ignore lint/a11y/useSemanticElements: searchable Popover + Command combobox.
          role="combobox"
          aria-label="Protocol"
          aria-expanded={open}
          className={cn("w-full justify-between font-normal", className)}
        >
          <span className="truncate">
            {selected?.name ??
              (loading ? "Loading protocols…" : (allLabel ?? "Choose a published protocol"))}
          </span>
          <ChevronsUpDown className="size-3.5 shrink-0 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[max(20rem,var(--radix-popover-trigger-width))] max-w-[calc(100vw-2rem)] p-0"
      >
        <Command
          filter={(_value, search, keywords) =>
            (keywords ?? []).join(" ").toLowerCase().includes(search.toLowerCase()) ? 1 : 0
          }
        >
          <CommandInput placeholder="Search protocols or targets…" />
          <CommandList>
            <CommandEmpty>{loading ? "Loading protocols…" : "No matching protocols."}</CommandEmpty>
            {allLabel && (
              <CommandItem
                value="all"
                keywords={[allLabel]}
                onSelect={() => {
                  onChange("");
                  setOpen(false);
                }}
              >
                {allLabel}
                {!value && <Check className="ml-auto size-4" />}
              </CommandItem>
            )}
            {protocols.map((protocol) => (
              <CommandItem
                key={protocol.id}
                value={protocol.id}
                keywords={[
                  protocol.name,
                  ...(protocol.readouts ?? []).map((readout) => readout.name),
                ]}
                onSelect={() => {
                  onChange(protocol.id);
                  setOpen(false);
                }}
              >
                <div className="min-w-0">
                  <p className="break-words">{protocol.name}</p>
                  {protocol.protocol_version != null && (
                    <p className="text-xs text-muted-foreground">
                      v{protocol.protocol_version} ·{" "}
                      {(protocol.readouts ?? []).filter((r) => r.type !== "probability").length}{" "}
                      targets
                    </p>
                  )}
                </div>
                {value === protocol.id && <Check className="ml-auto size-4 shrink-0" />}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
