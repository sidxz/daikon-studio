"use client";

import { type ReactNode, createContext, useContext } from "react";

export interface AppConfig {
  apiBaseUrl: string;
  appUrl: string;
  sentinelUrl: string;
  serviceName: string;
  idp: {
    googleClientId: string;
    entraClientId: string;
    entraTenantId: string;
  };
  uiVersion: string;
  uiGitSha: string;
  uiBuildDate: string;
  environment: string;
}

const defaultConfig: AppConfig = {
  apiBaseUrl: "http://localhost:8002",
  appUrl: "http://localhost:3003",
  sentinelUrl: "http://localhost:9003",
  serviceName: "daikon-studio",
  idp: { googleClientId: "", entraClientId: "", entraTenantId: "" },
  uiVersion: "0.0.0+dev",
  uiGitSha: "unknown",
  uiBuildDate: "unknown",
  environment: "development",
};

const AppConfigContext = createContext<AppConfig>(defaultConfig);

export function AppConfigProvider({
  config,
  children,
}: { config: AppConfig; children: ReactNode }) {
  return <AppConfigContext.Provider value={config}>{children}</AppConfigContext.Provider>;
}

export function useAppConfig(): AppConfig {
  return useContext(AppConfigContext);
}

/**
 * Runtime config from the server route. There is deliberately no NEXT_PUBLIC_
 * fallback: those are baked into the client bundle at build time, which is the
 * thing /api/config exists to avoid, and the fallback would only ever fire if a
 * Next app failed to serve its own route.
 */
export async function fetchAppConfig(): Promise<AppConfig> {
  try {
    const response = await fetch("/api/config");
    if (response.ok) return await response.json();
  } catch {
    // Unreachable during SSR and in tests -- typed defaults are the answer.
  }
  return defaultConfig;
}
