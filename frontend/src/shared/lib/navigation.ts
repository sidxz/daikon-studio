import {
  CollectionsIcon,
  DatasetsIcon,
  EnginesIcon,
  ProtocolsIcon,
  RunnersIcon,
  RunsIcon,
  SweepsIcon,
} from "@/shared/components/icons/nav-icons";
import { LayoutDashboard } from "lucide-react";
import type { ComponentType } from "react";

export interface NavItem {
  title: string;
  href: string;
  icon: ComponentType<{ className?: string }>;
  /** The item's hue, a `text-icon-*` utility. Applied to the icon only; labels stay neutral. */
  iconClassName: string;
  children?: NavItem[];
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

/**
 * One declaration drives three consumers: the sidebar, the ⌘K palette, and the
 * breadcrumb bar's labels and linkability.
 *
 * The grouping follows what a user is doing rather than what the API calls
 * things. "Curate" because the design opens by calling this a curator's
 * platform; Curate builds a Protocol, Apply runs someone else's.
 */
export const navigation: NavGroup[] = [
  {
    label: "Overview",
    items: [
      {
        title: "Dashboard",
        href: "/",
        icon: LayoutDashboard,
        iconClassName: "text-icon-dashboard",
      },
    ],
  },
  {
    label: "Curate",
    items: [
      {
        title: "Datasets",
        href: "/datasets",
        icon: DatasetsIcon,
        iconClassName: "text-icon-datasets",
      },
      {
        title: "Protocols",
        href: "/protocols",
        icon: ProtocolsIcon,
        iconClassName: "text-icon-protocols",
      },
      { title: "Sweeps", href: "/sweeps", icon: SweepsIcon, iconClassName: "text-icon-sweeps" },
    ],
  },
  {
    label: "Apply",
    items: [
      { title: "Runs", href: "/runs", icon: RunsIcon, iconClassName: "text-icon-runs" },
      {
        title: "Collections",
        href: "/collections",
        icon: CollectionsIcon,
        iconClassName: "text-icon-collections",
      },
    ],
  },
  {
    label: "Catalog",
    items: [
      { title: "Engines", href: "/engines", icon: EnginesIcon, iconClassName: "text-icon-engines" },
      { title: "Runners", href: "/runners", icon: RunnersIcon, iconClassName: "text-icon-runners" },
    ],
  },
];

/** Flattened, for breadcrumb label resolution and the palette. */
export const allNavItems: NavItem[] = navigation.flatMap((group) =>
  group.items.flatMap((item) => [item, ...(item.children ?? [])]),
);
