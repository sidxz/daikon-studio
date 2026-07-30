"use client";

import { LogoMark } from "@/shared/components/logo-mark";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { forgetWorkspace } from "@/shared/lib/auth/workspace-memory";
import { AuthzCallback } from "@sentinel-auth/nextjs";
import { useRouter } from "next/navigation";
import { WorkspaceSelector } from "./workspace-selector";

export default function AuthCallbackPage() {
  const router = useRouter();

  return (
    <div className="fixed inset-0 overflow-hidden bg-background">
      {/* Cover art, CSS and one SVG only -- no animation library for a page you
          see once a day. The mark bleeds off-canvas at low opacity so the panel
          keeps all the contrast. */}
      <div className="pointer-events-none absolute inset-0 hidden md:right-[460px] md:block">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_40%,var(--color-sidebar)_0%,transparent_70%)]" />
        <LogoMark className="absolute left-1/2 top-1/2 size-[min(80vh,780px)] -translate-x-1/2 -translate-y-1/2 opacity-[0.07]" />
      </div>
      <div className="relative z-20 flex min-h-screen flex-col md:ml-auto md:w-[460px] md:border-l md:border-sidebar-border md:bg-sidebar">
        <div
          className="flex flex-col items-end px-8 pt-8"
          style={{ animation: "auth-enter 0.7s ease-out 0.1s both" }}
        >
          <div className="flex items-center gap-3">
            <LogoMark className="size-12" />
            <span
              className="text-3xl font-medium tracking-tight"
              style={{ fontFamily: "var(--font-overused-grotesk), ui-sans-serif, sans-serif" }}
            >
              DAIKON Studio
            </span>
          </div>
        </div>

        <div className="flex flex-1 flex-col items-center justify-center">
          <div
            className="w-full max-w-[320px] px-6 md:px-0"
            style={{ animation: "auth-enter 0.7s ease-out 0.2s both" }}
          >
            <AuthzCallback
              onSuccess={(_user, returnTo) => router.replace(returnTo ?? "/")}
              onError={(error) => {
                // A failed auto-entry must not loop: forget the remembered
                // workspace so the next attempt shows the picker instead of
                // retrying the same broken one forever.
                forgetWorkspace();
                router.replace(`/login?error=${encodeURIComponent(error.message)}`);
              }}
              onSilentReauthFailed={() => router.replace("/login")}
              loadingComponent={
                <div>
                  <h2 className="text-sm font-medium text-muted-foreground">Signing in…</h2>
                  <div className="mt-4 space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                </div>
              }
              workspaceSelector={(props) => <WorkspaceSelector {...props} />}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
