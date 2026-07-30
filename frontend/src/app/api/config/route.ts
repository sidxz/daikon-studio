/**
 * Runtime configuration.
 *
 * Every value is read at *request* time, not build time, so one Docker image
 * serves every environment. That is why these are APP_-prefixed and read here
 * on the server rather than NEXT_PUBLIC_-prefixed and baked into the client
 * bundle.
 *
 * The build identity (uiVersion / uiGitSha / uiBuildDate) is image-specific and
 * genuinely is baked in, via Docker ARGs; it rides along here so the client has
 * one config fetch rather than two.
 */
export function GET() {
  return Response.json({
    apiBaseUrl: process.env.APP_API_BASE_URL ?? "http://localhost:8002",
    appUrl: process.env.APP_URL ?? "http://localhost:3003",
    sentinelUrl: process.env.APP_SENTINEL_URL ?? "http://localhost:9003",
    serviceName: process.env.APP_SENTINEL_SERVICE_NAME ?? "daikon-studio",
    idp: {
      googleClientId: process.env.APP_SENTINEL_GOOGLE_CLIENT_ID ?? "",
      entraClientId: process.env.APP_SENTINEL_ENTRA_CLIENT_ID ?? "",
      entraTenantId: process.env.APP_SENTINEL_ENTRA_TENANT_ID ?? "",
    },
    uiVersion: process.env.APP_VERSION || "0.0.0+dev",
    uiGitSha: process.env.APP_GIT_SHA || "unknown",
    uiBuildDate: process.env.APP_BUILD_DATE || "unknown",
    environment: process.env.APP_ENV || "development",
  });
}
