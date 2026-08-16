"use client";

import { LogoMark } from "@/shared/components/logo-mark";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarTrigger,
} from "@/shared/components/ui/sidebar";
import { useAuthz } from "@duar-auth/nextjs";
import { Settings } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AppVersionTag } from "./app-version-tag";
import { NavMain } from "./nav-main";

export function AppSidebar(props: React.ComponentProps<typeof Sidebar>) {
  const pathname = usePathname();
  const { user } = useAuthz();

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        {/* Static brand block: it shows the workspace, it is not a switcher.
            Switching lives in the header's account menu, because the IdP token
            is memory-only in authz mode and cannot list workspaces from here. */}
        <div className="flex items-center gap-2 px-2 py-1.5 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
          {/* Wrapper div: SidebarMenuButton clamps direct-child svgs to size-4. */}
          <div className="flex size-8 shrink-0 items-center justify-center">
            <LogoMark className="size-8" />
          </div>
          <div className="grid flex-1 text-left text-sm leading-tight group-data-[collapsible=icon]:hidden">
            <span
              className="truncate text-[15px] font-medium tracking-tight text-sidebar-text-active"
              style={{ fontFamily: "var(--font-overused-grotesk), ui-sans-serif, sans-serif" }}
            >
              DAIKON Studio
            </span>
            <span className="truncate text-xs uppercase tracking-widest text-sidebar-text opacity-60">
              {user?.workspaceSlug ?? ""}
            </span>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <NavMain />
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={pathname === "/settings"} tooltip="Settings">
              <Link href="/settings">
                <Settings />
                <span>Settings</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <div className="flex items-center justify-between border-t border-sidebar-border px-3 py-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
          <AppVersionTag />
          <SidebarTrigger className="size-7 text-sidebar-foreground/40 hover:text-sidebar-foreground" />
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
