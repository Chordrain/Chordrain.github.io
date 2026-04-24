#!/usr/bin/env bash
# Preview the built site *with* Cloudflare Pages Functions (likes, GitHub OAuth).
# Requires a previous `./scripts/build.sh`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d dist ]; then
  echo "▸ No dist/ found – building first…"
  ./scripts/build.sh
fi

echo "▸ Starting Wrangler (Pages) preview on http://localhost:8788 (Ctrl+C to stop)"
echo "  – Likes API:  /api/likes/<slug>"
echo "  – OAuth:      /api/auth/github/login  →  /api/auth/github/callback"
echo "  – Tip: set .dev.vars for GITHUB_CLIENT_SECRET / SESSION_SECRET (see README)."
exec npx wrangler pages dev ./dist \
  --compatibility-date=2024-11-01 \
  --port 8788 \
  "$@"
