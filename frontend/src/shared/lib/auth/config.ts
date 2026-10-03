import type { AppConfig } from "@/shared/lib/app-config";
import { AuthzLocalStorageStore, DuarAuthz, type IdpConfig, IdpConfigs } from "@duar-auth/js";

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

let _client: DuarAuthz | null = null;

/**
 * The one Duar client. Takes runtime `AppConfig` so no `process.env` is
 * read at module load -- that is what lets a single Docker image serve every
 * environment.
 */
export function getDuarClient(config?: AppConfig): DuarAuthz {
  if (!_client) {
    _client = new DuarAuthz({
      duarUrl: config?.duarUrl ?? "http://localhost:9003",
      idps: config ? buildIdps(config) : {},
      redirectUri: `${config?.appUrl ?? "http://localhost:3003"}/auth/callback`,
      // Required since Duar 0.11.0: the browser no longer mints authz
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
