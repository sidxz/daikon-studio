"use client";

import type { WorkspaceMember } from "@duar-auth/js";
import { useMemo } from "react";

import { getDuarClient } from "@/shared/lib/auth/config";
import { useWorkspaceMembers } from "@/shared/lib/auth/use-workspace-members";

import type { MentionItem } from "./editor/core";

export type Member = WorkspaceMember;

// ponytail: one roster fetch per session, not per keystroke; a member who joins
// mid-session shows up after a reload.
let roster: Promise<Member[]> | null = null;

/** `@` mention source: workspace members, filtered client-side (Duar lists them all). */
export async function searchMemberItems(query: string): Promise<MentionItem[]> {
  const q = query.trim().toLowerCase();
  roster ??= getDuarClient()
    .listMembers()
    .catch((err) => {
      roster = null;
      throw err;
    });
  const members = await roster;
  return members
    .filter((m) => !q || [m.name, m.email].some((s) => s?.toLowerCase().includes(q)))
    .slice(0, 10)
    .map((m) => ({
      id: m.user_id,
      label: m.name || m.email,
      email: m.email,
      avatarUrl: m.avatar_url,
    }));
}

/** Member's display label: name, falling back to email. */
export const displayName = (m: Member): string => m.name || m.email;

/** id → member lookup, memoized. */
export function useMemberIndex(): { byId: Map<string, Member>; isPending: boolean } {
  const { data, isPending } = useWorkspaceMembers();
  const byId = useMemo(() => new Map((data ?? []).map((m) => [m.user_id, m])), [data]);
  return { byId, isPending };
}
