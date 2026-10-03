"use client";

import { setUnauthorizedHandler } from "@/shared/lib/api/custom-instance";
import { getDuarClient, idpTokenExpiresAt } from "@/shared/lib/auth/config";
import { useEffect } from "react";

const RENEW_WINDOW_MS = 90_000;
const CHECK_EVERY_MS = 30_000;
// A fresh token lives an hour, so a 401 this soon after a renewal is not expiry
// (a misconfigured backend, say), and renewing again would redirect through the
// IdP forever. sessionStorage, because the renewal itself reloads the page.
const LOOP_GUARD_MS = 5 * 60_000;
const LAST_RENEWAL_KEY = "studio.lastSessionRenewal";

/** Pure: renew when the IdP token expires inside the window, or already has. */
export function shouldRenew(expiresAt: number | null, now: number): boolean {
  return expiresAt !== null && expiresAt - now < RENEW_WINDOW_MS;
}

/**
 * Google ID tokens live one hour and the implicit flow has no refresh token, so
 * the SDK can only renew by a prompt=none redirect through the IdP. Do that
 * ourselves shortly before expiry while the tab is visible, and on the first
 * 401 if we missed it. `silentLogin` records the current path and query itself
 * and the callback page returns there; in-memory page state does not survive.
 */
export function useSessionRenewal(): void {
  useEffect(() => {
    const renew = () => {
      if (document.visibilityState !== "visible") return;
      if (Date.now() - Number(sessionStorage.getItem(LAST_RENEWAL_KEY)) < LOOP_GUARD_MS) return;
      if (getDuarClient().silentLogin()) {
        sessionStorage.setItem(LAST_RENEWAL_KEY, String(Date.now()));
      }
    };
    setUnauthorizedHandler(renew);
    const timer = window.setInterval(() => {
      if (shouldRenew(idpTokenExpiresAt(), Date.now())) renew();
    }, CHECK_EVERY_MS);
    return () => {
      window.clearInterval(timer);
      setUnauthorizedHandler(null);
    };
  }, []);
}
