"use client";

import { Check } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

import { Input } from "@/shared/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";
import { cn } from "@/shared/lib/utils";

export type ComboboxOption = {
  value: string;
  /** Text used for sync filtering and as the display fallback. */
  label: string;
  /** Extra text folded into sync filtering (e.g. an email). */
  keywords?: string;
  /** Custom row content; falls back to `label`. */
  node?: ReactNode;
};

/** The one option-row recipe (D2) — shared with the tiptap mention list, which owns its own keyboard. */
export function comboboxOptionClass(state: { active?: boolean; selected?: boolean }): string {
  return cn(
    "flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-foreground outline-none",
    state.active ? "bg-accent text-accent-foreground" : "hover:bg-muted",
    state.selected && !state.active && "bg-muted",
  );
}

interface ComboboxProps {
  /** The trigger element (a Button); wrapped in PopoverTrigger asChild. */
  trigger: ReactNode;
  /** Static options (sync mode) — filtered client-side by the query. Ignored when `onSearch` is set. */
  options?: ComboboxOption[];
  /** Async source (async mode): debounced, called on open and on each query change. */
  onSearch?: (query: string) => Promise<ComboboxOption[]>;
  /** Selected values — drive `aria-selected` and the multi-select check mark. */
  selected?: string[];
  /** Multi-select toggle mode: the popover stays open and rows show a check. Single-select closes on pick. */
  multiple?: boolean;
  onSelect: (option: ComboboxOption) => void;
  searchPlaceholder?: string;
  searchAriaLabel?: string;
  emptyText?: string;
  loadingText?: string;
  /** Rendered below the list, inside the popover; receives a close() callback and
   *  the current search query (existing callers that only take `close` still typecheck). */
  footer?: (close: () => void, query: string) => ReactNode;
  /** Debounce for async `onSearch`. Tasks callers pass MEMBER_SEARCH_DEBOUNCE_MS. */
  searchDebounceMs?: number;
}

/**
 * Popover + search Input + listbox with real combobox semantics: `role="listbox"`/
 * `option`, `aria-selected`, ArrowUp/Down + Enter, debounced async search, sync
 * filtering, single- or multi-select. One primitive behind the member/label/access
 * pickers (D2).
 */
export function Combobox({
  trigger,
  options = [],
  onSearch,
  selected = [],
  multiple = false,
  onSelect,
  searchPlaceholder = "Search…",
  searchAriaLabel,
  emptyText = "No results.",
  loadingText = "Loading…",
  footer,
  searchDebounceMs = 250,
}: ComboboxProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [results, setResults] = useState<ComboboxOption[]>([]);
  const [loading, setLoading] = useState(false);
  // Async search rejected — surfaced distinctly from an empty result so a failed
  // fetch doesn't masquerade as "no matches". Reset on open and on next keystroke.
  const [failed, setFailed] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const baseId = useId();
  const listId = `${baseId}-listbox`;
  const optionId = (i: number) => `${baseId}-option-${i}`;

  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const list = useMemo(() => {
    if (onSearch) return results;
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter(
      (o) => o.label.toLowerCase().includes(q) || (o.keywords ?? "").toLowerCase().includes(q),
    );
  }, [onSearch, results, options, query]);

  // Async mode: debounce the query, then fetch — with a stale guard so a slow
  // earlier response can't overwrite a newer one. `loading` is raised in the
  // open/keystroke handlers (not here) so this effect stays free of sync setState.
  useEffect(() => {
    if (!onSearch || !open) return;
    let stale = false;
    const t = setTimeout(() => {
      onSearch(query).then(
        (r) => {
          if (!stale) {
            setResults(r);
            setLoading(false);
            setFailed(false);
          }
        },
        () => {
          if (!stale) {
            setResults([]);
            setLoading(false);
            setFailed(true);
          }
        },
      );
    }, searchDebounceMs);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [onSearch, open, query, searchDebounceMs]);

  const active = Math.min(activeIndex, Math.max(0, list.length - 1));

  // Keep the highlighted option in view during keyboard navigation.
  useEffect(() => {
    if (open) document.getElementById(optionId(active))?.scrollIntoView({ block: "nearest" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, open]);

  // Reset the search state on open (event handler, not an effect); raise `loading`
  // up front in async mode so the list shows "Loading…" instead of a brief "empty".
  const handleOpenChange = (next: boolean) => {
    if (next) {
      setQuery("");
      setActiveIndex(0);
      setFailed(false);
      if (onSearch) setLoading(true);
    }
    setOpen(next);
  };

  const close = () => setOpen(false);

  const commit = (opt: ComboboxOption | undefined) => {
    if (!opt) return;
    onSelect(opt);
    if (multiple) inputRef.current?.focus();
    else close();
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (list.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (Math.min(i, list.length - 1) + 1) % list.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (Math.min(i, list.length - 1) - 1 + list.length) % list.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      commit(list[active]);
    }
  };

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-0">
        <div className="p-2">
          <Input
            ref={inputRef}
            value={query}
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={list.length > 0 ? optionId(active) : undefined}
            placeholder={searchPlaceholder}
            aria-label={searchAriaLabel ?? searchPlaceholder}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
              setFailed(false);
              if (onSearch) setLoading(true);
            }}
            onKeyDown={onKeyDown}
          />
        </div>
        <div
          role="listbox"
          id={listId}
          aria-multiselectable={multiple || undefined}
          className="max-h-64 overflow-y-auto px-1 pb-1"
        >
          {loading ? (
            <p className="px-2 py-1.5 text-sm text-muted-foreground">{loadingText}</p>
          ) : failed ? (
            <p className="px-2 py-1.5 text-sm text-muted-foreground">Couldn’t load results.</p>
          ) : list.length === 0 ? (
            <p className="px-2 py-1.5 text-sm text-muted-foreground">{emptyText}</p>
          ) : (
            list.map((o, i) => {
              const isSelected = selectedSet.has(o.value);
              return (
                <div
                  key={o.value}
                  id={optionId(i)}
                  role="option"
                  aria-selected={isSelected}
                  onMouseEnter={() => setActiveIndex(i)}
                  onClick={() => commit(o)}
                  className={comboboxOptionClass({ active: i === active, selected: isSelected })}
                >
                  <span className="flex min-w-0 flex-1 items-center gap-2">{o.node ?? o.label}</span>
                  {multiple && isSelected && <Check className="size-4 shrink-0 text-foreground" />}
                </div>
              );
            })
          )}
        </div>
        {footer && <div className="border-t border-border p-2">{footer(close, query)}</div>}
      </PopoverContent>
    </Popover>
  );
}
