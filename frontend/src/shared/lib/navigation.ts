import {
  Boxes,
  Cpu,
  Database,
  FlaskConical,
  LayoutDashboard,
  type LucideIcon,
  PlayCircle,
  Server,
} from "lucide-react";

export interface NavItem {
  title: string;
  href: string;
  icon: LucideIcon;
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
    items: [{ title: "Dashboard", href: "/", icon: LayoutDashboard }],
  },
  {
    label: "Curate",
    items: [
      { title: "Datasets", href: "/datasets", icon: Database },
      { title: "Protocols", href: "/protocols", icon: FlaskConical },
    ],
  },
  {
    label: "Apply",
    items: [
      { title: "Runs", href: "/runs", icon: PlayCircle },
      { title: "Collections", href: "/collections", icon: Boxes },
    ],
  },
  {
    label: "Catalog",
    items: [
      { title: "Engines", href: "/engines", icon: Cpu },
      { title: "Runners", href: "/runners", icon: Server },
    ],
  },
];

/** Flattened, for breadcrumb label resolution and the palette. */
export const allNavItems: NavItem[] = navigation.flatMap((group) =>
  group.items.flatMap((item) => [item, ...(item.children ?? [])]),
);
