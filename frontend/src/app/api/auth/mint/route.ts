/**
 * Authz token mint (BFF).
 *
 * Since Sentinel 0.11.0 the browser no longer calls /authz/resolve itself --
 * that needs a service key, which must stay server-side. The SentinelAuthz
 * client POSTs the IdP token here, same-origin, and this route forwards it with
 * the key attached. Workspace discovery still goes browser -> Sentinel
 * directly; only credential issuance is proxied.
 */
export async function POST(request: Request) {
  const sentinelUrl = (process.env.APP_SENTINEL_URL ?? "http://localhost:9003").replace(/\/+$/, "");
  const serviceKey = process.env.APP_SENTINEL_SERVICE_KEY;

  if (!serviceKey) {
    return Response.json(
      { detail: "Mint endpoint not configured: APP_SENTINEL_SERVICE_KEY is missing." },
      { status: 503 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  const upstream = await fetch(`${sentinelUrl}/authz/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Service-Key": serviceKey },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  return Response.json(data, { status: upstream.status });
}
