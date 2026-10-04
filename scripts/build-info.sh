#!/usr/bin/env sh
# Single source of build identity for BOTH components, used by CI and `make dev`.
#   scripts/build-info.sh backend|frontend
# prints three KEY=VALUE lines: VERSION, GIT_SHA, BUILD_DATE.
# VERSION comes from the nearest `<component>-vX.Y.Z` tag via `git describe`
# ("0.1.0" on the tag, "0.1.0-3-gabc1234[-dirty]" past it); with no tag at all
# it is "0.0.0+dev" — the same fallback both apps use when nothing is baked in.
set -eu
component="${1:?usage: build-info.sh backend|frontend}"
if git describe --tags --match "${component}-v*" --abbrev=0 >/dev/null 2>&1; then
  version="$(git describe --tags --match "${component}-v*" --dirty | sed "s/^${component}-v//")"
else
  version="0.0.0+dev"
fi
echo "VERSION=${version}"
echo "GIT_SHA=$(git rev-parse --short=7 HEAD)"
echo "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
