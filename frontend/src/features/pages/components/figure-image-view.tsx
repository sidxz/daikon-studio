"use client";

import type { NodeViewProps } from "@tiptap/react";

import { useBlobObjectUrl } from "@/features/pages/lib/blobs";
import { Skeleton } from "@/shared/components/ui/skeleton";

import { EmbedFigure, type EmbedWidth } from "./embed-toolbar";

export function FigureImageView(props: NodeViewProps) {
  const { blobKey, alt, caption, width } = props.node.attrs as {
    blobKey: string | null;
    alt: string;
    caption: string | null;
    width: EmbedWidth | null;
  };
  const url = useBlobObjectUrl(blobKey);
  const size = width ?? "full";

  return (
    <EmbedFigure {...props} kind="figure" width={size}>
      {url ? (
        // src is a client-side blob: object URL (useBlobObjectUrl); next/image's
        // optimizer fetches from a URL and can't reach a blob: one, so there's no
        // next/image equivalent here.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt={alt} className="h-auto w-full rounded-md border border-border" />
      ) : (
        <Skeleton className="h-40 w-full rounded-md" />
      )}
      {caption ? (
        <figcaption className="mt-1 text-sm text-muted-foreground">{caption}</figcaption>
      ) : null}
    </EmbedFigure>
  );
}
