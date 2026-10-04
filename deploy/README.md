# Deploying daikon-studio

One host, one Compose stack: Caddy (TLS) in front of the API and the frontend,
Postgres, and a default-lane runner. The API migrates the database itself when it
starts. Every command below runs from this directory.

## 1. Prerequisites

1. A Linux host with Docker 24+ and the Compose plugin: `docker compose version`.
2. A DNS A record for your domain pointing at the host: `dig +short studio.example.edu`.
3. Ports 80 and 443 open to the internet. Caddy gets the TLS certificate over them.
4. The realm's Google OAuth client lists `https://<domain>/auth/callback` under
   Authorised redirect URIs.
5. A Duar Service App registered for this deployment. Its name goes in
   `*_DUAR_SERVICE_NAME`, its key in `*_DUAR_SERVICE_KEY`.
6. If the ghcr.io packages are private, log in with a token that has `read:packages`:
   `echo "$TOKEN" | docker login ghcr.io -u <github-user> --password-stdin`.

## 2. First boot

1. `cp .env.example .env`, then fill it in. `POSTGRES_PASSWORD`: `openssl rand -hex 24`.
   The Duar key and the Google client id each go in twice, once per prefix.
   Leave `STUDIO_RUNNER_TOKEN_DEFAULT` empty for now.
2. `docker compose pull`
3. `docker compose up -d caddy`. This starts everything except the runner: Caddy
   pulls in the API and frontend, and the API waits for Postgres, then migrates it
   before it serves. The runner needs a token from section 3; started without one it
   exits with code 2 and Compose restarts it until the token is set.
4. `docker compose logs -f api` until the API logs `Application startup complete`,
   after one `Running upgrade` line per migration.
   If it restarts instead, the traceback usually names Duar: an empty service key,
   or a `STUDIO_DUAR_URL` this host cannot reach (the API fetches Duar's signing key
   at boot and exits without it).
5. Open `https://<domain>` and sign in.

## 3. First runner

1. Sign in as a user with the admin role. Runners > Add runner, any name, the
   Default lane only. The dialog shows the token once, inside a `docker run` command.
2. Put the `drt_…` value into `.env` as `STUDIO_RUNNER_TOKEN_DEFAULT`.
3. `docker compose up -d runner-default`. This restarts the API too, since it reads
   the same `.env`.
4. `docker compose logs runner-default` shows `runner agent started`, and the
   Runners page shows it online.

A GPU runner (chemprop, MoLFormer) runs on its own x86_64 host with an NVIDIA GPU,
a driver for CUDA 12.x (525+; `nvidia-smi` shows the CUDA version it supports), and
the NVIDIA Container Toolkit:

5. The image is `ghcr.io/sidxz/daikon-studio/runner-gpu:<backend version>`, published
   with each backend release by `make publish-runner-gpu` (RELEASING.md), not by CI.
6. Add runner again, GPU lane only, and run the command the dialog prints there:
   `docker run -d --restart unless-stopped --gpus all -e STUDIO_URL=https://<domain> -e STUDIO_RUNNER_TOKEN=drt_… ghcr.io/sidxz/daikon-studio/runner-gpu:<version>`

That runner reaches the API through the reverse proxy, and prediction artifact
reads put a full `file:///…` URI in the request path. Caddy forwards it intact.
nginx's default `merge_slashes on` collapses it to `file:/…` and the read fails, so
an nginx in front of this stack needs `merge_slashes off;`.

## 4. Upgrade

1. `docker compose pull`
2. `docker compose up -d`. The new API applies any new migrations before it serves.
3. `docker compose ps`: api `(healthy)`.

Upgrading from a release before multi-target datasets (migration 013) needs an
order, because the runner and API wire shapes changed with no version handshake.
Before step 2, wait until no training run is running. Then deploy the API (which
migrates) and upgrade the runners, which `docker compose up -d` does in that order here
(every GPU runner on another host needs its image upgraded too). A training run in
flight across the API deploy ends FAILED beside a complete, unlinked protocol, and
a runner left on the old image fails every run it claims.

CI tags every image with the commit sha as well as `latest`. To pin a release, set
`STUDIO_API_IMAGE=ghcr.io/sidxz/daikon-studio/api:<sha>` and the matching frontend
tag. Migrations only go forward: rolling back past one means restoring a backup.

## 5. Backup and restore

The database first, then the blobs, so every blob a dumped row points at is in the
archive. The volume is `deploy_blobs` because Compose names the project after this
directory.

1. `docker compose exec -T postgres pg_dump -U studio studio | gzip > studio-$(date +%F).sql.gz`
2. `docker run --rm -v deploy_blobs:/blobs -v "$PWD:/out" alpine tar czf /out/blobs-$(date +%F).tgz -C /blobs .`
3. Nightly, from cron (crontab treats `%` as a newline, hence `\%`):

   ```
   30 2 * * * cd /srv/daikon-studio/deploy && docker compose exec -T postgres pg_dump -U studio studio | gzip > /var/backups/studio-$(date +\%F).sql.gz && docker run --rm -v deploy_blobs:/blobs -v /var/backups:/out alpine tar czf /out/blobs-$(date +\%F).tgz -C /blobs .
   ```

Restore, onto a booted stack (on a new host, run First boot first):

1. `docker compose stop caddy frontend runner-default api`
2. `docker compose exec -T postgres dropdb -U studio studio && docker compose exec -T postgres createdb -U studio studio`
3. `gunzip -c studio-YYYY-MM-DD.sql.gz | docker compose exec -T postgres psql -q -v ON_ERROR_STOP=1 -U studio -d studio`
4. `docker run --rm -v deploy_blobs:/blobs -v "$PWD:/in" alpine tar xzf /in/blobs-YYYY-MM-DD.tgz -C /blobs`.
   Blobs written after the backup stay on the volume, referenced by nothing.
5. `docker compose up -d`

## 6. Logs and health

1. `docker compose ps`: api `(healthy)`, the rest `Up`.
2. `docker compose logs -f api runner-default`. One JSON object per line.
3. `curl -s https://<domain>/ready`: `{"status":"ready"}`, or a 503 naming the
   database failure. `/health` answers whenever the process is up.
4. Every API log line carries a `request_id`, and every response returns it as
   `X-Request-ID`. From a browser repro, copy it from the failing request's response
   headers (DevTools > Network), then `docker compose logs api | grep <id>`.

## 7. What this stack does not do

1. No GPU runner on this host. Without one (section 3), gpu-lane runs stay pending.
2. No object storage. Blobs live on the `blobs` volume. For S3 or MinIO, before
   first boot, set `STUDIO_BLOB_BASE_URL=s3://…` in `compose.yml` (it overrides `.env`)
   and `STUDIO_BLOB_STORAGE_OPTIONS` in `.env`.
3. One API process. No horizontal scaling, and an upgrade restarts it.
4. `/docs` and `/openapi.json` are public. To hide them, drop both from the
   `@backend` matcher in the `Caddyfile` and `docker compose restart caddy`.
5. No log rotation. Docker's default `json-file` driver keeps everything; set
   `"log-driver": "local"` in `/etc/docker/daemon.json`.
- It is a **single trust domain**. Runner tokens are instance-wide (any workspace's runs can be
  claimed by any runner on the lane), so this stack is for one lab, or for labs that trust each
  other's runner machines. Separate labs get separate stacks.

## 7. Running migrations yourself

The API migrates on start under a Postgres advisory lock, so two APIs starting
together are safe. To make migrations a separate, deliberate step instead, set
`STUDIO_MIGRATE_ON_START=false` on the API and run, before each deploy:
`docker compose run --rm api alembic upgrade head`.
