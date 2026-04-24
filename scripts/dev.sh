#!/usr/bin/env bash
# Start the Astro dev server. Likes/auth APIs need Wrangler (see preview.sh);
# `astro dev` alone will serve everything *except* those Pages Functions.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d node_modules ]; then
  echo "▸ Installing dependencies…"
  npm install
fi

echo "▸ Starting Astro dev server on http://localhost:4321 (Ctrl+C to stop)"
exec npm run dev -- "$@"
