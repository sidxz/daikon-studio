"use client";

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/shared/components/ui/command";
import { navigation } from "@/shared/lib/navigation";
import { useCommandPaletteStore } from "@/shared/lib/stores/command-palette-store";
import { cn } from "@/shared/lib/utils";
import { FileUp, FlaskConical, Play } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect } from "react";

export function CommandPalette() {
  const open = useCommandPaletteStore((s) => s.open);
  const setOpen = useCommandPaletteStore((s) => s.setOpen);
  const toggle = useCommandPaletteStore((s) => s.toggle);
  const router = useRouter();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") {
        event.preventDefault();
        toggle();
      }
    },
    [toggle],
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Go to a page or action"
      description="Find a page or start a new task."
    >
      <CommandInput placeholder="Go to a page or action…" />
      <CommandList>
        <CommandEmpty>No matching pages or actions.</CommandEmpty>
        <CommandGroup heading="Create">
          {[
            { title: "Upload dataset", href: "/datasets/new", icon: FileUp },
            { title: "Train a protocol", href: "/protocols/new", icon: FlaskConical },
            { title: "Run predictions", href: "/runs/new", icon: Play },
          ].map((action) => (
            <CommandItem
              key={action.href}
              onSelect={() => {
                setOpen(false);
                router.push(action.href);
              }}
            >
              <action.icon aria-hidden className="mr-2 size-4" />
              {action.title}
            </CommandItem>
          ))}
        </CommandGroup>
        {navigation.map((group) => (
          <CommandGroup key={group.label} heading={group.label}>
            {group.items.map((item) => (
              <CommandItem
                key={item.href}
                onSelect={() => {
                  setOpen(false);
                  router.push(item.href);
                }}
              >
                <item.icon className={cn("mr-2 size-4", item.iconClassName)} />
                {item.title}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}
