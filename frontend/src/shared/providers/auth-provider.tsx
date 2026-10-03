"use client";

import { setApiBaseUrl } from "@/shared/lib/api/custom-instance";
import { type AppConfig, AppConfigProvider, fetchAppConfig } from "@/shared/lib/app-config";
import { shouldAutoReauth } from "@/shared/lib/auth/auto-reauth";
import { getDuarClient } from "@/shared/lib/auth/config";
import type { DuarAuthz } from "@duar-auth/js";
import { AuthzProvider } from "@duar-auth/nextjs";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const clientRef = useRef<DuarAuthz | null>(null);
  const configRef = useRef<AppConfig | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    let cancelled = false;

    fetchAppConfig()
      .then((config) => {
        if (cancelled) return;
        // Before anything can fetch. This is why QueryProvider nests inside
        // AuthProvider rather than beside it.
        setApiBaseUrl(config.apiBaseUrl);
        clientRef.current = getDuarClient(config);
        configRef.current = config;
        setMounted(true);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Could not load configuration");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-destructive">Configuration error</p>
          <p className="mt-1 text-sm text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  if (!mounted || !clientRef.current || !configRef.current) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading…</div>
      </div>
    );
  }

  return (
    <AppConfigProvider config={configRef.current}>
      <AuthzProvider client={clientRef.current} autoReauth={shouldAutoReauth(pathname)}>
        {children}
      </AuthzProvider>
    </AppConfigProvider>
  );
}
