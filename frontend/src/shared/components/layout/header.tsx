"use client";

import { Avatar, AvatarFallback } from "@/shared/components/ui/avatar";
import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import { Separator } from "@/shared/components/ui/separator";
import { forgetWorkspace } from "@/shared/lib/auth/workspace-memory";
import { useCommandPaletteStore } from "@/shared/lib/stores/command-palette-store";
import { useAuthz } from "@duar-auth/nextjs";
import { Building2, ChevronDown, LogOut, Search } from "lucide-react";
import { Breadcrumbs } from "./breadcrumbs";
import { FontSizeControl } from "./font-size-control";
import { ThemeToggle } from "./theme-toggle";

export function Header() {
  const { user, logout } = useAuthz();
  const openPalette = useCommandPaletteStore((s) => s.setOpen);

  const initials = user?.name
    ? user.name
        .split(" ")
        .map((part) => part[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
    : "?";

  return (
    <header className="flex h-10 shrink-0 items-center gap-2 border-b border-border/60 px-4">
      <Breadcrumbs />
      <div className="ml-auto flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => openPalette(true)}
          className="hidden gap-2 text-muted-foreground md:flex"
        >
          <Search className="size-4" />
          <span className="text-xs">Search</span>
          <kbd className="pointer-events-none ml-1 inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
            <span className="text-xs">⌘</span>K
          </kbd>
        </Button>
        <FontSizeControl />
        <ThemeToggle />
        <Separator orientation="vertical" className="mx-1.5 data-[orientation=vertical]:h-5" />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-auto gap-2 px-2 py-1">
              <span className="sr-only">Account menu</span>
              <Avatar className="size-7 rounded-lg">
                <AvatarFallback className="rounded-lg text-xs">{initials}</AvatarFallback>
              </Avatar>
              <div className="hidden flex-col text-left leading-tight sm:flex">
                <span className="text-xs font-medium">{user?.name ?? "User"}</span>
                <span className="text-[10px] text-muted-foreground">{user?.email ?? ""}</span>
              </div>
              <ChevronDown className="size-3.5 text-muted-foreground" aria-hidden />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <p className="truncate text-sm font-medium">{user?.name ?? "User"}</p>
              <p className="truncate text-xs font-normal text-muted-foreground">
                {user?.email ?? ""}
              </p>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => {
                // Forget first, so the next sign-in shows the picker rather
                // than dropping straight back into the workspace being left.
                forgetWorkspace();
                logout();
              }}
            >
              <Building2 />
              Switch workspace
            </DropdownMenuItem>
            <DropdownMenuItem variant="destructive" onSelect={() => logout()}>
              <LogOut />
              Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
