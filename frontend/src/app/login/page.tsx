"use client";

import { LogoMark } from "@/shared/components/logo-mark";
import { useAppConfig } from "@/shared/lib/app-config";
import { useAuthz } from "@duar-auth/nextjs";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";

function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden="true">
      <title>Google</title>
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.65l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.11a6.6 6.6 0 0 1 0-4.22V7.05H2.18a11 11 0 0 0 0 9.9l3.66-2.84z"
      />
      <path
        fill="#EA4335"
        d="M12 4.75c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 1.46 14.97.5 12 .5A11 11 0 0 0 2.18 7.05l3.66 2.84C6.71 7.29 9.14 4.75 12 4.75z"
      />
    </svg>
  );
}

/** A failed sign-in lands here as `?error=` (see the callback page); say what it was. */
function LoginError() {
  const error = useSearchParams().get("error");
  if (!error) return null;
  return (
    <p
      role="alert"
      className="mt-3 rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive"
    >
      {error}
    </p>
  );
}

/* Shared by the animated mark and its reduced-motion static twin. 28%/20%
   (light/dark) replaces the old 7%, which vanished on white. */
const coverMarkClass =
  "absolute left-1/2 top-1/2 size-[min(80vh,780px)] -translate-x-1/2 -translate-y-1/2 opacity-[0.28] dark:opacity-[0.20] [filter:drop-shadow(0_0_12px_rgba(96,130,255,0.2))]";

export default function LoginPage() {
  const { isAuthenticated, login } = useAuthz();
  const { idp } = useAppConfig();
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated) router.replace("/");
  }, [isAuthenticated, router]);

  const hasEntra = Boolean(idp.entraClientId && idp.entraTenantId);

  return (
    <div className="fixed inset-0 overflow-hidden bg-background">
      {/* Cover art, CSS and one SVG only -- no animation library for a page you
          see once a day. The suite spectrum orbits the ring (one SMIL element,
          7s per lap); reduced-motion users get the static twin via the CSS
          swap. Opacity and a same-hue bloom keep the mark legible on white. */}
      <div className="pointer-events-none absolute inset-0 hidden md:right-[460px] md:block">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_40%,var(--color-sidebar)_0%,transparent_70%)]" />
        <LogoMark animate className={`${coverMarkClass} motion-reduce:hidden`} />
        <LogoMark className={`${coverMarkClass} hidden motion-reduce:block`} />
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
          <p className="mt-1 text-xs text-muted-foreground">
            In-silico protocols for drug discovery
          </p>
        </div>

        <div className="flex flex-1 flex-col items-center justify-center">
          <div className="w-full max-w-[320px] px-6 md:px-0">
            <div style={{ animation: "auth-enter 0.7s ease-out 0.2s both" }}>
              <h2 className="text-sm font-medium text-muted-foreground">Sign in to continue</h2>
              {/* useSearchParams needs a suspense boundary in the app router. */}
              <Suspense fallback={null}>
                <LoginError />
              </Suspense>
              <button
                type="button"
                onClick={() => login("google")}
                className="mt-4 flex w-full cursor-pointer items-center justify-center gap-3 rounded-xl bg-white px-4 py-2.5 text-sm font-medium text-gray-800 transition-all duration-200 hover:-translate-y-px active:translate-y-0"
              >
                <GoogleIcon />
                Continue with Google
              </button>
              {hasEntra && (
                <button
                  type="button"
                  onClick={() => login("entraId")}
                  className="mt-2 flex w-full cursor-pointer items-center justify-center gap-3 rounded-xl border border-border px-4 py-2.5 text-sm font-medium transition-all duration-200 hover:-translate-y-px active:translate-y-0"
                >
                  Continue with Microsoft
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
