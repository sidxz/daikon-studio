import type { AppConfig } from "@/shared/lib/app-config";
import {
  AuthzLocalStorageStore,
  type IdpConfig,
  IdpConfigs,
  SentinelAuthz,
} from "@sentinel-auth/js";

function buildIdps(config: AppConfig): Record<string, IdpConfig> {
  const idps: Record<string, IdpConfig> = {};
  if (config.idp.googleClientId) {
    idps.google = IdpConfigs.google(config.idp.googleClientId);
  }
  if (config.idp.entraClientId && config.idp.entraTenantId) {
    idps.entraId = IdpConfigs.entraId(config.idp.entraClientId, config.idp.entraTenantId);
  }
  return idps;
}

let _client: SentinelAuthz | null = null;

/**
 * The one Sentinel client. Takes runtime `AppConfig` so no `process.env` is
 * read at module load -- that is what lets a single Docker image serve every
 * environment.
 */
export function getSentinelClient(config?: AppConfig): SentinelAuthz {
  if (!_client) {
    _client = new SentinelAuthz({
      sentinelUrl: config?.sentinelUrl ?? "http://localhost:9003",
      idps: config ? buildIdps(config) : {},
      redirectUri: `${config?.appUrl ?? "http://localhost:3003"}/auth/callback`,
      // Required since Sentinel 0.11.0: the browser no longer mints authz
      // tokens directly, because minting needs a service key that must stay
      // server-side. It POSTs to this same-origin route instead.
      mintEndpoint: "/api/auth/mint",
      storage: typeof window !== "undefined" ? new AuthzLocalStorageStore() : undefined,
      autoRefresh: true,
      refreshBuffer: 30,
    });
  }
  return _client;
}
