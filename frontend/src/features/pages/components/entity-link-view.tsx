"use client";

import { type NodeViewProps, NodeViewWrapper } from "@tiptap/react";
import Link from "next/link";

// Studio entities a `#` chip can point at, each with its own detail route.
const ROUTES: Record<string, string> = {
  dataset: "/datasets",
  protocol: "/protocols",
  run: "/runs",
};

/**
 * The `#` chip: a frozen label deep-linked to the entity's detail page. Hover
 * preview is minimal — label + type via `title`.
 */
export function EntityLinkView({ node }: NodeViewProps) {
  const { entityType, entityId, label } = node.attrs as {
    entityType: string;
    entityId: string;
    label: string;
  };
  const route = ROUTES[entityType];

  return (
    <NodeViewWrapper as="span" className="mention">
      {route ? (
        <Link href={`${route}/${entityId}`} title={`${label} (${entityType})`}>
          {label}
        </Link>
      ) : (
        <span title={`${label} (${entityType})`}>{label}</span>
      )}
    </NodeViewWrapper>
  );
}
