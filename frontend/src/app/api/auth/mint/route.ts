/**
 * Authz token mint (BFF).
 *
 * Since Duar 0.11.0 the browser no longer calls /authz/resolve itself --
 * that needs a service key, which must stay server-side. The DuarAuthz
 * client POSTs the IdP token here, same-origin, and this route forwards it with
 * the key attached. Workspace discovery still goes browser -> Duar
 * directly; only credential issuance is proxied.
 */
export async function POST(request: Request) {
  const duarUrl = (process.env.APP_DUAR_URL ?? "http://localhost:9003").replace(/\/+$/, "");
  const serviceKey = process.env.APP_DUAR_SERVICE_KEY;

  if (!serviceKey) {
    return Response.json(
      { detail: "Mint endpoint not configured: APP_DUAR_SERVICE_KEY is missing." },
      { status: 503 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  const upstream = await fetch(`${duarUrl}/authz/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Service-Key": serviceKey },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  return Response.json(data, { status: upstream.status });
}
