# syntax=docker/dockerfile:1
# Next.js standalone build. Runtime config is read from APP_* env at request
# time (src/app/api/config/route.ts), so one image serves every environment.

FROM node:22-alpine AS build
RUN corepack enable && corepack prepare pnpm@11.0.8 --activate
WORKDIR /app
# Lockfile and the allowBuilds approvals first, so a source change does not re-resolve.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# The RDKit wasm reaches public/ through a postinstall hook that is `|| true` --
# in a clean image that copy ran before public/ existed and failed silently,
# and every structure then renders blank. Copy it where a missing file is a
# build failure instead.
RUN cp node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm public/RDKit_minimal.wasm
RUN pnpm build

FROM node:22-alpine AS runtime
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
# Build identity is read at request time by /api/config, so it is baked as ENV.
# Declared in this stage only: in the build stage a fresh APP_BUILD_DATE would
# bust the `pnpm build` cache on every build.
ARG APP_VERSION=0.0.0+dev
ARG APP_GIT_SHA=unknown
ARG APP_BUILD_DATE=unknown
ENV APP_VERSION=$APP_VERSION APP_GIT_SHA=$APP_GIT_SHA APP_BUILD_DATE=$APP_BUILD_DATE
WORKDIR /app
# The runtime needs node only. npm, corepack and pnpm come with the base image
# and carry their own dependency trees, which is where most image CVEs land.
RUN rm -rf /usr/local/lib/node_modules /usr/local/bin/npm /usr/local/bin/npx \
  /usr/local/bin/corepack /usr/local/bin/pnpm /usr/local/bin/pnpx
RUN addgroup -S app && adduser -S app -G app
# Standalone output omits the two directories the server expects beside it.
COPY --from=build --chown=app:app /app/.next/standalone ./
COPY --from=build --chown=app:app /app/.next/static ./.next/static
COPY --from=build --chown=app:app /app/public ./public
USER app
EXPOSE 3000
CMD ["node", "server.js"]
