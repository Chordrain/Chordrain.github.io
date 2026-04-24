#!/usr/bin/env bash
# Deploy the site to Cloudflare Pages.
# Requires a one-time `npx wrangler login`.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_NAME="${CF_PAGES_PROJECT:-caeastro-blog}"

./scripts/build.sh

echo "▸ Deploying to Cloudflare Pages project: ${PROJECT_NAME}"
exec npx wrangler pages deploy ./dist --project-name "${PROJECT_NAME}" "$@"
