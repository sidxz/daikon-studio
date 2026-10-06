"use client";

import { FontSizeControl } from "@/shared/components/layout/font-size-control";
import { LogoMark } from "@/shared/components/logo-mark";
import { PageHeader } from "@/shared/components/page-header";
import { Button } from "@/shared/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { useApiVersion } from "@/shared/hooks/use-api-version";
import { useAppConfig } from "@/shared/lib/app-config";
import { type FontFamily, useFontFamilyStore } from "@/shared/lib/stores/font-family-store";
import { useTheme } from "next-themes";

const THEMES = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
] as const;

const FONTS: { value: FontFamily; label: string }[] = [
  { value: "plex", label: "IBM Plex" },
  { value: "inter", label: "Inter" },
  { value: "merriweather", label: "Merriweather" },
];

function VersionRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-xs">{value}</span>
    </div>
  );
}

export default function SettingsPage() {
  const { resolvedTheme, setTheme } = useTheme();
  const font = useFontFamilyStore((s) => s.font);
  const setFont = useFontFamilyStore((s) => s.setFont);
  const config = useAppConfig();
  const api = useApiVersion();

  return (
    <div className="w-full min-w-0 space-y-4">
      <PageHeader title="Settings" description="Reading preferences and application information." />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Appearance</CardTitle>
          <p className="text-xs text-muted-foreground">
            Per-user preferences, stored in this browser.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm">Text size</span>
            <FontSizeControl />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Theme</span>
            <div className="flex gap-1">
              {THEMES.map((option) => (
                <Button
                  key={option.value}
                  size="sm"
                  variant={resolvedTheme === option.value ? "secondary" : "ghost"}
                  aria-pressed={resolvedTheme === option.value}
                  onClick={() => setTheme(option.value)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Font</span>
            <div className="flex gap-1">
              {FONTS.map((option) => (
                <Button
                  key={option.value}
                  size="sm"
                  variant={font === option.value ? "secondary" : "ghost"}
                  aria-pressed={font === option.value}
                  onClick={() => setFont(option.value)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <LogoMark className="size-5" />
            About DAIKON Studio
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="mb-1 text-xs font-medium">Interface</p>
            <VersionRow label="Version" value={config.uiVersion} />
            <VersionRow label="Commit" value={config.uiGitSha} />
            <VersionRow label="Built" value={config.uiBuildDate} />
          </div>
          <div>
            <p className="mb-1 text-xs font-medium">API</p>
            {api.isLoading ? (
              <p className="text-xs text-muted-foreground">Loading…</p>
            ) : api.isError ? (
              <p className="text-xs text-muted-foreground">Unavailable</p>
            ) : (
              <>
                <VersionRow label="Version" value={api.data?.version ?? "unknown"} />
                <VersionRow label="Commit" value={api.data?.git_sha ?? "unknown"} />
                <VersionRow label="Built" value={api.data?.build_date ?? "unknown"} />
              </>
            )}
          </div>
          <VersionRow label="Environment" value={config.environment} />
        </CardContent>
      </Card>
    </div>
  );
}
