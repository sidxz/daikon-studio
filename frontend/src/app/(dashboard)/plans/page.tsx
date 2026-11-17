"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/**
 * Planning moved under Studies; old links still land. A client-side replace
 * rather than a server redirect, because a server redirect arrives as a fresh
 * document request and the sign-in state does not travel with it, which sent
 * the old link to the login page.
 */
export default function PlansPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/studies/new");
  }, [router]);
  return null;
}
