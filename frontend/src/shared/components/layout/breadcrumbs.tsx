"use client";

import { allNavItems } from "@/shared/lib/navigation";
import {
  useBreadcrumbOverrides,
  useBreadcrumbTrailValue,
} from "@/shared/lib/stores/breadcrumb-store";
import { ChevronRight, Home } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const UUID_SEGMENT = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Hrefs that are real pages, so an intermediate path segment is not linked. */
const linkableHrefs = new Set(allNavItems.map((item) => item.href));

/**
 * Three tiers, in precedence order: a trail a page declared for itself, a
 * per-segment override, then the URL.
 *
 * The URL fallback is the dangerous one -- a detail route's segment is a UUID,
 * and printing it raw breaks the rule that a user never sees an identifier. So
 * every detail page declares a trail; this fallback exists for the list pages,
 * whose segments are all real words.
 */
export function Breadcrumbs() {
  const pathname = usePathname();
  const overrides = useBreadcrumbOverrides();
  const trail = useBreadcrumbTrailValue();
  const segments = pathname.split("/").filter(Boolean);

  if (segments.length === 0) {
    return <span className="text-sm font-medium">Dashboard</span>;
  }

  // Guarded on the last crumb having a label so a still-loading entity does not
  // flash an empty crumb before its name arrives.
  if (trail && trail.length > 0 && trail[trail.length - 1].label) {
    return (
      <nav className="flex items-center gap-1 text-sm" aria-label="Breadcrumb">
        <Link href="/" className="text-muted-foreground transition-colors hover:text-foreground">
          <Home className="size-3.5" />
        </Link>
        {trail.map((crumb, index) => {
          const isLast = index === trail.length - 1;
          return (
            <span key={crumb.href ?? crumb.label} className="flex items-center gap-1">
              <ChevronRight className="size-3 text-muted-foreground" />
              {!isLast && crumb.href ? (
                <Link
                  href={crumb.href}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  {crumb.label}
                </Link>
              ) : (
                <span className={isLast ? "font-medium" : "text-muted-foreground"}>
                  {crumb.label}
                </span>
              )}
            </span>
          );
        })}
      </nav>
    );
  }

  function resolveLabel(href: string, segment: string): string {
    const override = overrides.get(segment);
    if (override) return override;
    const match = allNavItems.find((item) => item.href === href);
    if (match) return match.title;
    // A detail page that failed to load never declares its trail, so its UUID
    // segment lands here. Never print it (no UUIDs on screen); say what it is.
    if (UUID_SEGMENT.test(segment)) return "Details";
    return segment.charAt(0).toUpperCase() + segment.slice(1);
  }

  return (
    <nav className="flex items-center gap-1 text-sm" aria-label="Breadcrumb">
      <Link href="/" className="text-muted-foreground transition-colors hover:text-foreground">
        <Home className="size-3.5" />
      </Link>
      {segments.map((segment, index) => {
        const href = `/${segments.slice(0, index + 1).join("/")}`;
        const isLast = index === segments.length - 1;
        return (
          <span key={href} className="flex items-center gap-1">
            <ChevronRight className="size-3 text-muted-foreground" />
            {!isLast && linkableHrefs.has(href) ? (
              <Link
                href={href}
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                {resolveLabel(href, segment)}
              </Link>
            ) : (
              <span className={isLast ? "font-medium" : "text-muted-foreground"}>
                {resolveLabel(href, segment)}
              </span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
