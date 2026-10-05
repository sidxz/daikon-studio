"use client";

import { Button } from "@/shared/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/shared/components/ui/command";
import { Label } from "@/shared/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import type { ChemCellarImportResponse } from "@/shared/lib/api/model";
import { ChevronsUpDown } from "lucide-react";
import { useState } from "react";
import {
  useChemCellarProtocols,
  useChemCellarRuns,
  useImportChemCellarRun,
} from "../hooks/use-chemcellar";
import { runOption } from "../lib/chemcellar-runs";

/**
 * Pick a ChemCellar protocol and run; the server imports the run's compounds and
 * hands back an upload the wizard treats like a dropped file.
 */
export function ChemCellarPicker({
  onImported,
}: { onImported: (imported: ChemCellarImportResponse) => void }) {
  const [protocolId, setProtocolId] = useState("");
  const [protocolOpen, setProtocolOpen] = useState(false);
  const [runId, setRunId] = useState("");

  const protocols = useChemCellarProtocols();
  const runs = useChemCellarRuns(protocolId);
  const importRun = useImportChemCellarRun();

  async function pickRun(id: string) {
    setRunId(id);
    try {
      onImported(await importRun.mutateAsync(id));
    } catch {
      // The global mutation handler already surfaced the message.
      setRunId("");
    }
  }

  if (protocols.isError) {
    return <p className="text-sm text-destructive">{protocols.error.message}</p>;
  }

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <Label htmlFor="cellar-protocol">ChemCellar protocol</Label>
        <Popover open={protocolOpen} onOpenChange={setProtocolOpen}>
          <PopoverTrigger asChild>
            <Button
              id="cellar-protocol"
              variant="outline"
              // biome-ignore lint/a11y/useSemanticElements: the standard Popover + Command combobox; a <select> cannot host a search box.
              role="combobox"
              aria-expanded={protocolOpen}
              className="w-full justify-between font-normal"
            >
              <span className={protocolId ? "" : "text-muted-foreground"}>
                {protocols.data?.find((protocol) => protocol.id === protocolId)?.name ??
                  (protocols.isLoading ? "Loading…" : "Choose a protocol")}
              </span>
              <ChevronsUpDown className="size-4 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-(--radix-popover-trigger-width) p-0" align="start">
            {/* Substring, not cmdk's fuzzy default: with ~100 long assay names
                fuzzy matching lets nearly everything through. */}
            <Command
              filter={(_value, search, keywords) =>
                (keywords ?? []).join(" ").toLowerCase().includes(search.toLowerCase()) ? 1 : 0
              }
            >
              <CommandInput placeholder="Search protocols…" />
              <CommandList>
                <CommandEmpty>No protocol matches.</CommandEmpty>
                {(protocols.data ?? []).map((protocol) => (
                  <CommandItem
                    key={protocol.id}
                    value={protocol.id}
                    keywords={[protocol.name]}
                    onSelect={() => {
                      setProtocolId(protocol.id);
                      setRunId("");
                      setProtocolOpen(false);
                    }}
                  >
                    {protocol.name}
                  </CommandItem>
                ))}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
        {protocols.data && protocols.data.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No ChemCellar protocols in this workspace.
          </p>
        )}
      </div>

      {protocolId && (
        <div className="space-y-1.5">
          <Label htmlFor="cellar-run">Run</Label>
          <Select value={runId} onValueChange={pickRun} disabled={importRun.isPending}>
            <SelectTrigger id="cellar-run">
              <SelectValue placeholder={runs.isLoading ? "Loading…" : "Choose a run"} />
            </SelectTrigger>
            <SelectContent>
              {(runs.data ?? []).map((run) => {
                const option = runOption(run);
                return (
                  <SelectItem key={run.id} value={run.id} disabled={option.disabled}>
                    {option.label}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          {runs.data && runs.data.length === 0 && (
            <p className="text-xs text-muted-foreground">This protocol has no runs.</p>
          )}
          {importRun.isPending && (
            <p className="text-xs text-muted-foreground">Loading compounds…</p>
          )}
        </div>
      )}
    </div>
  );
}
