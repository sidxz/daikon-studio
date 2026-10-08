import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type { DatasetResponse, ProtocolResponse, RunResponse } from "@/shared/lib/api/model";

import type { MentionItem } from "./editor/core";

type Paged<T> = { items: T[] };

const PER_KIND = 5;

const list = <T>(path: string, q: string) =>
  customInstance<Paged<T>>({
    url: `${API_V1}/${path}`,
    method: "GET",
    params: { q: q || undefined, limit: PER_KIND },
  })
    .then((p) => p.items)
    .catch(() => [] as T[]);

/**
 * `#` mention source: the studio's own datasets, protocols and runs, searched by
 * name through each list route's `q`. A kind that fails to load just contributes
 * nothing, so one outage never blanks the menu.
 */
export async function searchEntities(query: string): Promise<MentionItem[]> {
  const q = query.trim();
  const [datasets, protocols, runs] = await Promise.all([
    list<DatasetResponse>("datasets", q),
    list<ProtocolResponse>("protocols", q),
    list<RunResponse>("runs", q),
  ]);
  return [
    ...datasets.map((d) => ({ id: d.id, label: d.name, kind: "dataset" })),
    ...protocols.map((p) => ({ id: p.id, label: p.name, kind: "protocol" })),
    ...runs.map((r) => ({ id: r.id, label: r.name ?? `Run ${r.id.slice(0, 8)}`, kind: "run" })),
  ];
}
