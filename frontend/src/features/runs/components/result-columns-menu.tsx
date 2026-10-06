import { Button } from "@/shared/components/ui/button";
import { Checkbox } from "@/shared/components/ui/checkbox";
import { Input } from "@/shared/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import { Columns3 } from "lucide-react";
import { useId, useState } from "react";

export interface ColumnChoice {
  id: string;
  label: string;
  shown: boolean;
  unavailable?: boolean;
}

export function ResultColumnsMenu({
  groups,
  onChange,
}: {
  groups: { id: string; name: string; columns: ColumnChoice[] }[];
  onChange: (choices: Record<string, boolean>) => void;
}) {
  const [search, setSearch] = useState("");
  const id = useId();
  const visible = groups.filter((group) =>
    `${group.name} ${group.columns.map((c) => c.label).join(" ")}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm">
          <Columns3 className="size-4" />
          Targets & columns
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <div className="border-b p-3">
          <Input
            aria-label="Search targets and columns"
            placeholder="Search targets and columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="max-h-96 overflow-y-auto p-3">
          {visible.length === 0 && (
            <p className="p-2 text-sm text-muted-foreground">No matching columns.</p>
          )}
          {visible.map((group) => {
            const available = group.columns.filter((column) => !column.unavailable);
            const count = available.filter((column) => column.shown).length;
            return (
              <fieldset key={group.id} className="mb-4 last:mb-0">
                <legend className="sr-only">{group.name}</legend>
                <label
                  htmlFor={`${id}-group-${group.id}`}
                  className="mb-2 flex items-center gap-2 text-sm font-medium"
                >
                  <Checkbox
                    id={`${id}-group-${group.id}`}
                    checked={
                      count === available.length && count > 0
                        ? true
                        : count > 0
                          ? "indeterminate"
                          : false
                    }
                    disabled={available.length === 0}
                    onCheckedChange={(checked) =>
                      onChange(
                        Object.fromEntries(
                          available.map((column) => [column.id, checked === true]),
                        ),
                      )
                    }
                  />
                  <span className="break-all">{group.name}</span>
                </label>
                <div className="space-y-2 pl-6">
                  {group.columns.map((column) => (
                    <label
                      key={column.id}
                      htmlFor={`${id}-column-${column.id}`}
                      className="flex items-center gap-2 text-sm"
                    >
                      <Checkbox
                        id={`${id}-column-${column.id}`}
                        checked={column.shown}
                        disabled={column.unavailable}
                        onCheckedChange={(checked) => onChange({ [column.id]: checked === true })}
                      />
                      <span className="min-w-0 break-words">
                        {column.label}
                        {column.unavailable && (
                          <span className="block text-xs text-muted-foreground">
                            Not reported in this run
                          </span>
                        )}
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>
            );
          })}
        </div>
      </PopoverContent>
    </Popover>
  );
}
