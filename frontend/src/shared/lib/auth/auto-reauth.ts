/**
 * Whether AuthzProvider's silent (`prompt=none`) re-auth should run on a route.
 *
 * It must not run on the auth-flow routes: on `/auth/callback` it would preempt
 * the OAuth response that page exists to process, and on `/login` it would
 * hijack an interactive sign-in. A stale-but-valid authz token keeps the SDK in
 * `needs_reauth` on every route, and an interactive login sets no in-flight
 * marker to guard against it.
 */
const AUTH_FLOW_PREFIXES = ["/login", "/auth"];

export function shouldAutoReauth(pathname: string | null): boolean {
  if (!pathname) return false;
  return !AUTH_FLOW_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}
