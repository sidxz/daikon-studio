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
import { SidebarTrigger } from "@/shared/components/ui/sidebar";
import { forgetWorkspace } from "@/shared/lib/auth/workspace-memory";
import { useCommandPaletteStore } from "@/shared/lib/stores/command-palette-store";
import { useAuthz } from "@duar-auth/nextjs";
import { Building2, ChevronDown, LogOut, Search } from "lucide-react";
import { Breadcrumbs } from "./breadcrumbs";
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
    <header className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-background px-3 sm:px-6 lg:px-8">
      <SidebarTrigger className="size-10 shrink-0 md:hidden" aria-label="Open navigation" />
      <div className="min-w-0 flex-1">
        <Breadcrumbs />
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => openPalette(true)}
          aria-label="Go to a page or action"
          className="size-10 gap-2 text-muted-foreground md:h-9 md:w-auto"
        >
          <Search className="size-4" />
          <span className="hidden text-sm md:inline">Go to…</span>
          <kbd className="pointer-events-none ml-1 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-xs font-medium text-muted-foreground md:inline-flex">
            <span className="text-xs">⌘</span>K
          </kbd>
        </Button>
        <ThemeToggle />
        <Separator orientation="vertical" className="mx-1.5 data-[orientation=vertical]:h-5" />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-auto gap-2 px-2 py-1">
              <span className="sr-only">Account menu</span>
              <Avatar className="size-7 rounded-lg">
                <AvatarFallback className="rounded-lg text-xs">{initials}</AvatarFallback>
              </Avatar>
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
