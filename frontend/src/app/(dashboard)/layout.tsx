"use client";

import { AppSidebar } from "@/shared/components/layout/app-sidebar";
import { Header } from "@/shared/components/layout/header";
import { SidebarInset, SidebarProvider } from "@/shared/components/ui/sidebar";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useAuthz } from "@sentinel-auth/nextjs";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

function DashboardSkeleton() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md space-y-4 px-4">
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
      </div>
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { authState, isLoading } = useAuthz();
  const router = useRouter();

  useEffect(() => {
    // Only bounce when the session is genuinely gone. `authState` is three-way,
    // and during `needs_reauth` the provider's own autoReauth is mid-redirect --
    // testing `!isAuthenticated` here would pre-empt it and force a full login.
    if (!isLoading && authState === "unauthenticated") {
      router.replace("/login");
    }
  }, [isLoading, authState, router]);

  if (isLoading || authState !== "authenticated") {
    return <DashboardSkeleton />;
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <Header />
        <main className="flex-1 overflow-auto p-4">{children}</main>
      </SidebarInset>
    </SidebarProvider>
  );
}
