import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import { shortDate } from "@/shared/lib/format-date";
import { cn } from "@/shared/lib/utils";
import { ChevronDown } from "lucide-react";
import { useId, useState } from "react";
import { localDay, parseLocalDay } from "../lib/run-date-range";

export function RunDateFilter({
  from,
  to,
  onChange,
}: {
  from: string;
  to: string;
  onChange: (from: string, to: string) => void;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState(from);
  const [end, setEnd] = useState(to);
  const invalid = Boolean(
    (start && !parseLocalDay(start)) ||
      (end && !parseLocalDay(end)) ||
      (start && end && start > end),
  );
  const startDate = parseLocalDay(from);
  const endDate = parseLocalDay(to);
  const label =
    startDate && endDate
      ? `${shortDate(startDate.toISOString())} – ${shortDate(endDate.toISOString())}`
      : startDate
        ? `Since ${shortDate(startDate.toISOString())}`
        : endDate
          ? `Through ${shortDate(endDate.toISOString())}`
          : "Date range";
  const apply = (from: string, to: string) => {
    onChange(from, to);
    setOpen(false);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(value) => {
        if (value) {
          setStart(from);
          setEnd(to);
        }
        setOpen(value);
      }}
    >
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          aria-label={`Date range: ${label}`}
          className={cn(
            "h-9 min-w-32 flex-1 justify-between font-normal sm:flex-none",
            (from || to) && "border-primary/30 bg-primary/5",
          )}
        >
          <span className="truncate">{label}</span>
          <ChevronDown className="size-3.5 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 max-w-[calc(100vw-2rem)] space-y-4">
        <div>
          <h2 className="text-sm font-semibold">Run dates</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Filter by when runs started. Both dates are included.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[
            { label: "Today", days: 1 },
            { label: "Last 7 days", days: 7 },
            { label: "Last 30 days", days: 30 },
          ].map(({ label, days }) => (
            <Button
              key={days}
              variant="secondary"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => {
                const today = new Date();
                const first = new Date(today);
                first.setDate(first.getDate() - days + 1);
                apply(localDay(first), localDay(today));
              }}
            >
              {label}
            </Button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="min-w-0 space-y-1.5">
            <Label htmlFor={`${id}-from`}>From</Label>
            <Input
              id={`${id}-from`}
              type="date"
              className="min-w-0"
              value={start}
              onChange={(event) => setStart(event.target.value)}
              aria-invalid={invalid}
            />
          </div>
          <div className="min-w-0 space-y-1.5">
            <Label htmlFor={`${id}-to`}>To</Label>
            <Input
              id={`${id}-to`}
              type="date"
              className="min-w-0"
              value={end}
              onChange={(event) => setEnd(event.target.value)}
              aria-invalid={invalid}
            />
          </div>
        </div>
        {invalid && (
          <p role="alert" className="text-xs text-destructive">
            Choose valid dates with the end on or after the start.
          </p>
        )}
        <div className="flex items-center justify-between">
          <Button variant="ghost" size="sm" onClick={() => apply("", "")}>
            All dates
          </Button>
          <Button size="sm" disabled={invalid} onClick={() => apply(start, end)}>
            Apply dates
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
